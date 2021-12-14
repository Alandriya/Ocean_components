import time
from calculations import *
from video import *
from struct import unpack
import multiprocessing
from multiprocessing import Pool
import os

# Parameters
files_path_prefix = 'D://Data/OceanFull/'
shift = 4 * 7
components_amount = 4
cpu_count = 8
window_EM = 200
start = 0
end = 29141
flux_type = 'sensible'
timesteps = 7320


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

            column_list = [f'mean_{i}' for i in range(1, components_amount + 1)] + \
                          [f'sigma_{i}' for i in range(1, components_amount + 1)] + \
                          [f'weight_{i}' for i in range(1, components_amount + 1)]
            new_data = pd.DataFrame(data=np.concatenate((means, sigmas, weights), axis=1), columns=column_list)
            new_data['ts'] = ts[:-shift:shift]
            new_data.to_csv(files_path_prefix + f'5_years_weekly/{flux_type}_{i}.csv', sep=';', index=False)

    print(f'Process {os.getpid()} finished')
    return


def sort_by_means(files_path_prefix, flux_type, mask):
    # load and sort Dataframes
    filename = os.listdir(files_path_prefix + '5_years_weekly/')[0]
    data = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')
    means_cols = data.filter(regex='mean_', axis=1).columns
    sigmas_cols = data.filter(regex='sigma_', axis=1).columns
    weights_cols = data.filter(regex='weight_', axis=1).columns

    for filename in tqdm.tqdm(os.listdir(files_path_prefix + '5_years_weekly/')):
        if flux_type in filename:
            df = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')

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

            df.to_csv(files_path_prefix + '5_years_weekly/' + filename, sep=';', index=False)
    return


def make_video(files_path_prefix, flux_type, mask):
    """

    :param files_path_prefix:
    :param flux_type:
    :return:
    """
    dataframes = list()
    indexes = list()
    for filename in tqdm.tqdm(os.listdir(files_path_prefix + '5_years_weekly/')):
        if flux_type in filename:
            df = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')
            dataframes.append(df)
            idx = int(filename[len(flux_type) + 1: -4])
            indexes.append(idx)

    tmp = list(zip(indexes, dataframes))
    tmp.sort()
    indexes = [y for y, _ in tmp]
    dataframes = [x for _, x in tmp]

    # draw pictures
    init_directory(files_path_prefix, flux_type)
    draw_frames(files_path_prefix, flux_type, mask, components_amount, dataframes, indexes)

    # create video from them
    # create_video(files_path_prefix, flux_type, f'{flux_type}_5years_weekly')
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
    delta = (end - start + cpu_count // 2) // cpu_count
    print(end - start)

    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    # ---------------------------------------------------------------------------------------
    # binary_to_array("l79-21", flux_type)

    # # ---------------------------------------------------------------------------------------
    # Calculations part

    # # print("Number of cpu : ", multiprocessing.cpu_count()) # 16 cpu
    # num_lost = []
    # for i in range(20335):
    #     if not os.path.exists(files_path_prefix + f'5_years_weekly/{flux_type}_{i}.csv'):
    #         num_lost.append(i)
    # print(len(num_lost))
    #
    # start = None
    # borders = []
    # sum_lost=0
    # for j in range(1, len(num_lost)):
    #     if start is None and mask[num_lost[j]]:
    #         start = num_lost[j]
    #
    #     # if (num_lost[j-1] == num_lost[j] - 1) and j != len(num_lost) - 1 and mask[j]:
    #     #     pass
    #     if not start is None and mask[num_lost[j]] and (num_lost[j-1] != num_lost[j] - 1):
    #         if j == len(num_lost) - 1:  # the end of the array
    #             borders.append([start, num_lost[j]])
    #         else:
    #             borders.append([start, num_lost[j-1]])
    #             sum_lost += num_lost[j-1] - start
    #             start = num_lost[j]
    #
    # # print(borders)
    # # print(sum_lost)
    # # raise ValueError

    # ts_array = np.load(files_path_prefix + f'5years_{flux_type}.npy')
    #
    # borders = [[start + delta*i, start + delta*(i+1)] for i in range(cpu_count)]
    #
    # masks = [mask[b[0]:b[1]] for b in borders]
    # ts_arrays = [ts_array[b[0]:b[1]] for b in borders]
    # args = [[borders[i], masks[i], ts_arrays[i]] for i in range(cpu_count)]
    #
    # print(borders)
    # print(len(borders))
    #
    # del ts_array, borders, masks, ts_arrays
    # with Pool(cpu_count) as p:
    #     p.map(cycle_part, args)

    # ---------------------------------------------------------------------------------------
    # # Components determination part
    # sort_by_means(files_path_prefix, flux_type, mask)
    # ---------------------------------------------------------------------------------------
    make_video(files_path_prefix, flux_type, mask)
    # create_video(files_path_prefix, flux_type, f'{flux_type}_5years_weekly', speed=30)

