import time
from calculations import *
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


if __name__ == '__main__':
    # filename = "l79-21"
    # length = 7320
    # arr_5years = np.empty((length, 29141), dtype=float)
    # for i in tqdm.tqdm(range(length)):
    #   file = open(files_path_prefix + filename, "rb")
    #   offset=32 + (62396 - length)*116564 + 116564*i
    #   file.seek(offset, 0)
    #   binary_values = file.read(116564)
    #   file.close()
    #   point = unpack('f'*29141, binary_values)
    #   arr_5years[i] = point
    # np.save(files_path_prefix + '5years_latent.npy', arr_5years.transpose())
    # del arr_5years

    # print("Number of cpu : ", multiprocessing.cpu_count()) # 16 cpu

    cpu_count = 8
    window_EM = 200
    shift = 4 * 7
    start = 2600
    end = 5384
    delta = (end-start) // cpu_count
    print(end - start)

    # num_lost = []
    # for i in range(20335):
    #     if not os.path.exists(files_path_prefix + f'5_years_weekly/sensible_{i}.csv'):
    #         num_lost.append(i)
    # print(len(num_lost))
    #
    # start = num_lost[0]
    # borders = []
    # for j in range(1, len(num_lost)):
    #     if (num_lost[j-1] == num_lost[j] - 1) and j != len(num_lost) - 1:
    #         pass
    #     else:
    #         if j == len(num_lost) - 1:
    #             borders.append([start, num_lost[j]])
    #         else:
    #             borders.append([start, num_lost[j-1]])
    #             start = num_lost[j]
    #
    # print(borders)
    # print(len(borders))
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


