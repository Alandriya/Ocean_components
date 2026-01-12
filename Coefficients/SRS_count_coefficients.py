import datetime
import os
from multiprocessing import Pool
from struct import unpack
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from EM_hybrid import *
from Kor_Bel_compare import plot_difference_1d
from Plotting.video import get_continuous_cmap

width = 181
height = 161


def count_abf_Kor_from_points(files_path_prefix: str, time_start: int, time_end: int, point_start: int, point_end: int):
    components = 3
    sens_df_list = list()
    sens_idxes = list()

    n_components = 3
    means_cols = [f'mean_{i}' for i in range(1, n_components + 1)]
    sigmas_cols = [f'sigma_{i}' for i in range(1, n_components + 1)]
    weights_cols = [f'weight_{i}' for i in range(1, n_components + 1)]

    print('Loading data')
    for p in tqdm.tqdm(range(point_start, point_end)):
        if os.path.exists(files_path_prefix + f'Components/sensible/raw/point_{p}.xlsx'):
            raw_sens = pd.read_excel(files_path_prefix + f'Components/sensible/raw/point_{p}.xlsx')
            raw_sens.columns = ['time', 'ts'] + means_cols + sigmas_cols + weights_cols
            sens_df_list.append(raw_sens)
            sens_idxes.append(p)

    for t in tqdm.tqdm(range(time_start, time_end)):
        a_sens = np.zeros((height, width))
        # a_lat = np.zeros((height, width))
        # b_matrix = np.zeros((4, height, width))
        # f = np.zeros((height, width), dtype=float)

        for i in range(len(sens_idxes)):
            p = sens_idxes[i]
            df = sens_df_list[i]
            a_sens[p // width, p % width] = sum([df.loc[t, f'mean_{comp}'] * df.loc[t, f'weight_{comp}']
                                                 for comp in range(1, components + 1)])
            # b_matrix[0, p // width, p % width] = None

        np.save(files_path_prefix + f'Components/sensible/{t}_A_sens.npy', a_sens)
    return


def process_points(files_path_prefix: str,
                   time_info: list,
                   points_info: list,
                   samples: np.ndarray,
                   flux_type: str,
                   coeff_type: str,
                   window_width: int,
                   step_ticks: int,
                   ticks_by_day: int,
                   radius: int,
                   n_components: int = 3,
                   draw: bool = False):
    print('My process id:', os.getpid())
    points_amount = len(points_info)
    time_start, time_end, timedelta = time_info
    for p_idx in range(points_amount):
        point, point_bigger, point_size = points_info[p_idx]
        sample = samples[p_idx]

        if not os.path.exists(files_path_prefix + f'Components/{flux_type}/raw'):
            os.mkdir(files_path_prefix + f'Components/{flux_type}/raw')

        if not os.path.exists(files_path_prefix + f'Components/{flux_type}/plots'):
            os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')

        if not os.path.exists(files_path_prefix + f'Components/{flux_type}/components-xlsx'):
            os.mkdir(files_path_prefix + f'Components/{flux_type}/components-xlsx')

        if True or not os.path.exists(
                files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx'):
            # apply EM
            point_df = hybrid(sample, window_width * point_size, n_components, EM_steps=1, step=step_ticks * point_size)
            point_df.to_excel(files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx',
                              index=False)

            df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx')
            new_df, new_n_components = cluster_components(df,
                                                          n_components,
                                                          files_path_prefix,
                                                          draw,
                                                          path=f'Components/{flux_type}/plots/',
                                                          postfix=f'_point_({point[0]}, {point[1]})-{coeff_type}')

            new_df.to_excel(
                files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx',
                index=False)

            Kor_df = new_df
            Kor_df.fillna(0, inplace=True)
            a_sum = np.zeros(len(Kor_df))
            b_sum = np.zeros(len(Kor_df))

            for i in range(1, n_components + 1):
                a_sum += Kor_df[f'mean_{i}'] * Kor_df[f'weight_{i}']

            for i in range(1, n_components + 1):
                b_sum += Kor_df[f'weight_{i}'] * (np.square(Kor_df[f'mean_{i}']) + np.square(Kor_df[f'sigma_{i}']))

            b_sum = np.sqrt(b_sum - np.square(a_sum))

            a_sum = np.array(a_sum[:len(a_sum) - len(a_sum) % (ticks_by_day // step_ticks)])
            a_sum = np.mean(a_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
            b_sum = np.array(b_sum[:len(b_sum) - len(b_sum) % (ticks_by_day // step_ticks)])
            b_sum = np.mean(b_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)

            np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy', a_sum)
            np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy', b_sum)
            if draw:
                plot_components(new_df,
                                new_n_components,
                                point,
                                files_path_prefix,
                                path=f'Components/{flux_type}/plots/',
                                postfix=f'_point_({point[0]}, {point[1]})-{coeff_type}')

                plot_a_sigma(df,
                             n_components,
                             point,
                             files_path_prefix,
                             path=f'Components/{flux_type}/plots/',
                             postfix=f'_point_({point[0]}, {point[1]})-{coeff_type}')

        if True or draw:
            plot_difference_1d(files_path_prefix,
                               time_start,
                               time_end,
                               point,
                               window_width,
                               radius,
                               ticks_by_day,
                               step_ticks,
                               n_components,
                               flux_type,
                               coeff_type)

    print(f'Process {os.getpid()} finished')
    return


def parallel_Korolev(files_path_prefix: str,
                     data_array: np.ndarray,
                     cpu_count: int,
                     points_borders: list,
                     time_start: int,
                     time_end: int,
                     timedelta: int,
                     flux_type: str,
                     coeff_type: str,
                     window_width: int,
                     step_ticks: int,
                     ticks_by_day: int,
                     radius: int,
                     n_components: int,
                     draw: bool = False):
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)

    # select points to list
    all_points = list()
    for p in range(points_borders[0], points_borders[1]):
        if mask[p]:
            all_points.append(p)

    # TODO delete
    from Plotting.plot_fluxes import plot_typical_points
    points = plot_typical_points(files_path_prefix, mask)
    all_points = [p[0] * width + p[1] for p in points]

    process_amount = len(all_points) // cpu_count + len(all_points) % cpu_count

    print(f'Processing {len(all_points)} points')
    # create args list
    all_args = list()
    for p in range(cpu_count):
        points = all_points[p * process_amount:(p + 1) * process_amount]
        points_info = list()
        samples = list()
        for point_idx in points:
            point_size = (radius * 2 + 1) ** 2
            point = (point_idx // width, point_idx % width)
            point_bigger = list()
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if 0 <= point[0] + i < height and 0 <= point[1] + j < width and \
                            mask[(point[0] + i) * width + point[1] + j]:
                        point_bigger.append((point[0] + i, point[1] + j))
                    else:
                        point_size -= 1

            sample = np.zeros((point_size, (time_end - time_start) * ticks_by_day - 1))
            for i in range(point_size):
                p = point_bigger[i]
                sample[i, :] = np.diff(
                    data_array[p[0] * width + p[1], time_start * ticks_by_day:time_end * ticks_by_day])

            # reshape
            sample = sample.transpose().flatten()

            points_info.append([point, point_bigger, point_size])
            samples.append(sample)

        time_info = [time_start, time_end, timedelta]
        args = [files_path_prefix, time_info, points_info, samples, flux_type, coeff_type, window_width, step_ticks,
                ticks_by_day, radius, n_components, draw]
        all_args.append(args)

    if cpu_count == 1:
        print("I'm the only process here")
        process_points(*all_args[0])
    else:
        print(f"I'm the parent with id = {os.getpid()}")
        with Pool(cpu_count) as p:
            p.starmap(process_points, all_args)
            p.close()
            p.join()
    return


def collect_EM(files_path_prefix: str,
               time_start: int,
               time_end: int,
               flux_type: str,
               coeff_type: str,
               folder: str = 'Sum',
               ):
    map_array = np.zeros((time_end - time_start, height, width), dtype=float)
    for i in tqdm.tqdm(range(height)):
        for j in range(width):
            point = (i, j)
            try:
                point_arr = np.load(files_path_prefix + f'Components/{flux_type}/{folder}/point_({point[0]}, {point[1]})-{coeff_type}.npy')
                map_array[:len(point_arr), i, j] = point_arr
            except FileNotFoundError:
                map_array[:, i, j] = np.nan
            except ValueError:
                print(f'Value error point ({i}, {j})')
        if i % 10 == 0:
            np.save(files_path_prefix + f'Components/{flux_type}/{coeff_type}_{time_start}-{time_end}_{folder}.npy',
                    map_array)
    np.save(files_path_prefix + f'Components/{flux_type}/{coeff_type}_{time_start}-{time_end}_{folder}.npy', map_array)
    return


def plot_map_Kor(files_path_prefix: str,
                 time_start: int,
                 time_end: int,
                 flux_type: str,
                 coeff_type: str,
                 radius: int,
                 start_pic_num: int = 0):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    img_values, img_difference = None, None

    data_array = np.load(files_path_prefix + f'Components/{flux_type}/{coeff_type}_{time_start}-{time_end}_Sum.npy')
    # diff_array = np.load(files_path_prefix + f'Components/{flux_type}/{coeff_type}_{time_start}-{time_end}_difference.npy')
    # data_array *= 2

    axs[0].set_title(f'{coeff_type} coeff Kor - {flux_type}', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax_values = divider.append_axes('right', size='5%', pad=0.3)

    axs[1].set_title(f'Bel-Kor difference', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax_difference = divider.append_axes('right', size='5%', pad=0.3)

    coeff_min = np.nanmin(data_array)
    coeff_max = np.nanmax(data_array)
    # print(coeff_min)
    # print(coeff_max)

    # diff_min = np.nanmin(diff_array)
    # diff_max = np.nanmax(diff_array)

    if coeff_type == 'A':
        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                   [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
        cmap.set_bad('darkgreen', 1.0)
        # cmap_diff = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - diff_min) / (diff_max - diff_min), 1])
        # cmap_diff.set_bad('darkgreen', 1.0)
    else:
        cmap = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
        cmap.set_bad('darkgreen', 1.0)
        # cmap_diff = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - diff_min) / (diff_max - diff_min), 1])
        # cmap_diff.set_bad('darkgreen', 1.0)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end - 1)):
        date = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        fig.suptitle(f'{coeff_type} coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        if flux_type == 'sensible':
            a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
        else:
            a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_lat.npy')

        if flux_type == 'sensible':
            b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[0]
        else:
            b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[3]

        if coeff_type == 'A':
            Bel_arr = a_arr
        else:
            Bel_arr = b_arr

        if img_values is None:
            img_values = axs[0].imshow(data_array[t],
                                       interpolation='none',
                                       cmap=cmap,
                                       vmin=coeff_min,
                                       vmax=coeff_max)
        else:
            img_values.set_data(data_array[t])

        fig.colorbar(img_values, cax=cax_values, orientation='vertical')

        diff_array = Bel_arr - data_array[t]
        diff_min = np.nanmin(diff_array)
        diff_max = np.nanmax(diff_array)
        cmap_diff = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                        [0, (1.0 - diff_min) / (diff_max - diff_min), 1])
        cmap_diff.set_bad('darkgreen', 1.0)

        if img_difference is None:
            img_difference = axs[1].imshow(diff_array,
                                           interpolation='none',
                                           cmap=cmap_diff,
                                           vmin=diff_min,
                                           vmax=diff_max)
        else:
            img_difference.set_data(diff_array)

        fig.colorbar(img_difference, cax=cax_difference, orientation='vertical')
        fig.savefig(files_path_prefix + f'Components/plots/{coeff_type}_{flux_type}/{pic_num:05d}.png')
        pic_num += 1
    return


