import time
from calculations import *
from video import *
from struct import unpack
import multiprocessing
from multiprocessing import Pool
import os

files_path_prefix = 'D://Data/Ocean-full/'
shift = 4 * 7


def cycle_part(arg):
    print('My process id:', os.getpid())
    borders, mask, ts_array = arg
    start, end = borders
    for i in range(start, end):
        if mask[i-start]:
            # print(f'Processing {i}-th point')
            ts = np.diff(ts_array[i-start])
            # time_start = time.time()
            means, sigmas, weights = apply_EM(ts, shift)
            # print(f'Elapsed {(time.time() - time_start) // 60} minutes {(time.time() - time_start) % 60} seconds == {time.time() - time_start} seconds')

            column_list = [f'mean_{i}' for i in range(1, 5)] + [f'sigma_{i}' for i in range(1, 5)] + [f'weight_{i}' for
                                                                                                      i in range(1, 5)]
            new_data = pd.DataFrame(data=np.concatenate((means, sigmas, weights), axis=1), columns=column_list)
            new_data['ts'] = ts[:-shift:shift]
            new_data.to_csv(files_path_prefix + f'5_years_weekly/sensible_{i}.csv', sep=';', index=False)
            # new_data.to_csv(files_path_prefix + f'5_years_weekly/latent_{i}.csv', sep=';', index=False)
    return


def make_video(files_path_prefix, flux_type):
    """

    :param files_path_prefix:
    :param flux_type:
    :return:
    """

    # load and sort Dataframes
    filename = os.listdir(files_path_prefix + '5_years_weekly/')[0]
    data = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')
    means_cols = data.filter(regex='mean_', axis=1).columns
    sigmas_cols = data.filter(regex='sigma_', axis=1).columns
    weights_cols = data.filter(regex='weight_', axis=1).columns

    dataframes = list()
    indexes = list()
    for filename in tqdm.tqdm(os.listdir(files_path_prefix + '5_years_weekly/5_years_weekly')):
        if flux_type in filename:
            df = pd.read_csv(files_path_prefix + '5_years_weekly/5_years_weekly/' + filename, delimiter=';')

            # sort all columns by means
            means = df[means_cols].values
            sigmas = df[sigmas_cols].values
            weights = df[weights_cols].values

            df.columns = list(means_cols) + list(sigmas_cols) + list(weights_cols) + ['ts']
            for i in range(len(df)):
                zipped = list(zip(means[i], sigmas[i], weights[i]))
                zipped.sort(key=lambda x: x[0])
                # the scary expression below is for flattening the sorted zip results
                df.iloc[i] = list(sum(list(zip(*zipped)), ())) + [df.loc[i, 'ts']]

            df.to_csv(files_path_prefix + '5_years_weekly/5_years_weekly/' + filename, sep=';', index=False)
            dataframes.append(df)
            idx = int(filename[len(flux_type) + 1: -4])
            indexes.append(idx)

    # tmp = list(zip(indexes, dataframes))
    # tmp.sort()
    # indexes = [y for y, _ in tmp]
    # dataframes = [x for _, x in tmp]

    # create video
    init_directory(files_path_prefix, flux_type)
    draw_pictures(files_path_prefix, flux_type, mask, components_amount, dataframes, indexes)
    return


def binary_to_array(filename, flux_type):
    #create array from binary data

    length = 7320
    arr_5years = np.empty((length, 29141), dtype=float)
    for i in tqdm.tqdm(range(length)):
      file = open(files_path_prefix + filename, "rb")
      offset=32 + (62396 - length)*116564 + 116564*i
      file.seek(offset, 0)
      binary_values = file.read(116564)
      file.close()
      point = unpack('f'*29141, binary_values)
      arr_5years[i] = point
    np.save(files_path_prefix + f'5years_{flux_type}.npy', arr_5years.transpose())
    del arr_5years
    return


if __name__ == '__main__':
    # Parameters
    cpu_count = 8
    components_amount = 4
    window_EM = 200
    shift = 4 * 7
    start = 0
    end = 20335
    delta = (end-start) // cpu_count
    print(end - start)
    flux_type = 'sensible'
    # ---------------------------------------------------------------------------------------
    # binary_to_array("l79-21", flux_type)

    # ---------------------------------------------------------------------------------------
    # Calculations part

    # print("Number of cpu : ", multiprocessing.cpu_count()) # 16 cpu
    num_lost = []
    for i in range(20335):
        if not os.path.exists(files_path_prefix + f'5_years_weekly/{flux_type}_{i}.csv'):
            num_lost.append(i)
    print(len(num_lost))

    start = num_lost[0]
    borders = []
    for j in range(1, len(num_lost)):
        if (num_lost[j-1] == num_lost[j] - 1) and j != len(num_lost) - 1:
            pass
        else:
            if j == len(num_lost) - 1:
                borders.append([start, num_lost[j]])
            else:
                borders.append([start, num_lost[j-1]])
                start = num_lost[j]

    print(borders)
    print(len(borders))
    # raise ValueError

    ts_array = np.load(files_path_prefix + '5years_sensible.npy')
    # ts_array = np.load(files_path_prefix + '5years_latent.npy')
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    borders = [[start + delta*i, start + delta*(i+1)] for i in range(cpu_count)]
    masks = [mask[b[0]:b[1]] for b in borders]
    ts_arrays = [ts_array[b[0]:b[1]] for b in borders]
    args = [[borders[i], masks[i], ts_arrays[i]] for i in range(cpu_count)]
    del ts_array, borders, masks, ts_arrays
    with Pool(cpu_count) as p:
        p.map(cycle_part, args)

    # for j in range(len(borders) // cpu_count):
    #     print(j)
    #     args = [[borders[j*cpu_count + i], masks[j*cpu_count + i], ts_arrays[j*cpu_count + i]] for i in range(cpu_count)]
    #     time.sleep(100)
    #     with Pool(cpu_count) as p:
    #         p.map(cycle_part, args)

    # ---------------------------------------------------------------------------------------
    # make_video(files_path_prefix, flux_type)



