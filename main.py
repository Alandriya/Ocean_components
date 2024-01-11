from Plotting.plot_Bel_coefficients import *
from SRS_count_coefficients import *
from Plotting.mean_year import *
from Plotting.video import *
from extreme_evolution import *
from ABCF_coeff_counting import *
from eigenvalues import *
from Plotting.plot_eigenvalues import plot_eigenvals

# Parameters
# files_path_prefix = 'D://Data/OceanFull/'
from data_processing import load_ABCFE

files_path_prefix = 'E:/Nastya/Data/OceanFull/'
# files_path_prefix = 'D://Data/OceanFull/'

# timesteps = 7320
timesteps = 1829
width = 181
height = 161

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2023, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # start_year = 1979
    # end_year = 2023
    # mask = mask.reshape((height, width))
    # for flux_type in ['sst']:
    #     for coeff_type in ['A']:
    #         # count_mean_year(files_path_prefix, start_year, end_year, coeff_type, flux_type, 'Bel', mask)
    #         mean_year = np.load(files_path_prefix + f'Mean_year/Bel/{flux_type}_{coeff_type}_{start_year}-{end_year}.npy')
    #         plot_mean_year_1d(files_path_prefix, mean_year, start_year, end_year, coeff_type, flux_type, 'Bel')
    # ----------------------------------------------------------------------------------------------
    # create videos
    # coeff_name = 'B'
    # create_video(files_path_prefix, f'videos/3D/{pair_name}/{coeff_name}/', f'{coeff_name}_',
    #              f'{pair_name}_{coeff_name}_2019-2023',
    #              start=days_delta1 + days_delta2 + days_delta3 + days_delta4 + 2)
    # ----------------------------------------------------------------------------------------------
    # count and plot extreme of coefficients 3d
    # pair_name = 'flux-sst'
    # pair_name = 'flux-press'
    # pair_name = 'sst-press'
    #
    # mean_days = 365
    # time_start = days_delta1 + days_delta2 + 1
    # time_end = time_start + days_delta3 - 1

    # if pair_name == 'flux-sst':
    #     names = ('Flux', 'SST')
    # elif pair_name == 'flux-press':
    #     names = ('Flux', 'Pressure')
    # else:
    #     names = ('SST', 'Pressure')

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, e_timelist, borders = load_ABCFE(files_path_prefix,
    #                                                                                  time_start,
    #                                                                                  time_end,
    #                                                                                  load_a=True,
    #                                                                                  load_b=True,
    #                                                                                  path_local=f'Coeff_data_3d/{pair_name}')
    # local_path_prefix = f'{pair_name}/'
    # extract_extreme(files_path_prefix, a_timelist, 'a', time_start, time_end, mean_days, local_path_prefix)
    # extract_extreme(files_path_prefix, b_timelist, 'b', time_start, time_end, mean_days,local_path_prefix)
    # plot_extreme(files_path_prefix, 'a', time_start, time_end, mean_days, local_path_prefix, names)
    # plot_extreme(files_path_prefix, 'b', time_start, time_end, mean_days, local_path_prefix, names)

    # coeff_type = 'b'
    # collect_extreme(files_path_prefix, coeff_type, local_path_prefix, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, 1, 16071, mean_days, local_path_prefix, names, fit_regression=True)
    # ----------------------------------------------------------------------------------------------
    # eigenvalues
    days_delta6 = (datetime.datetime(2022, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # pair_name = 'flux-sst'
    # pair_name = 'flux-press'
    # pair_name = 'sst-press'
    pair_name = 'sensible_latent'
    if pair_name == 'flux-sst':
        names = ('Flux', 'SST')
    elif pair_name == 'flux-press':
        names = ('Flux', 'Pressure')
    elif pair_name == 'sst-press':
        names = ('SST', 'Pressure')
    else:
        names = ('Sensible', 'Latent')
    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    # flux = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2023_grouped.npy')
    # sst = np.load(files_path_prefix + f'SST/SST_2019-2023_grouped.npy')
    # press = np.load(files_path_prefix + f'Pressure/PRESS_2019-2023_grouped.npy')
    #
    # sensible = np.load(files_path_prefix + f'Fluxes/sensible_grouped_1979-1989.npy')
    # latent = np.load(files_path_prefix + f'Fluxes/latent_grouped_1979-1989.npy')
    # count_abfe_coefficients(files_path_prefix, mask, sensible, latent, 0, sensible.shape[1]-1, offset=offset, pair_name=pair_name)

    time_start = offset + 1
    time_end = time_start + days_delta6 - 1
    a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, e_timelist, borders = load_ABCFE(files_path_prefix,
                                                                                                  time_start,
                                                                                                  time_end,
                                                                                                  load_e=True,
                                                                                                  path_local=f'Coeff_data/')

    count_eigenvalues(files_path_prefix, e_timelist, time_start, pair_name)

    lambda_timelist = list()
    lambda_min = [0, 0]
    lambda_max = [0, 0]

    for t in range(time_start, time_end):
        lambda_matrix = np.load(files_path_prefix + f'Eigenvalues/{pair_name}/{t}_lambda.npy')
        for i in range(2):
            lambda_max[i] = max(lambda_max[i], np.nanmax(lambda_matrix[i]))
            lambda_min[i] = min(lambda_min[i], np.nanmin(lambda_matrix[i]))
        lambda_timelist.append(lambda_matrix)

    pair_name = 'sensible-latent'
    path_local = f'{pair_name}/'
    plot_eigenvals(files_path_prefix, lambda_timelist, [lambda_max, lambda_min], 0, len(lambda_timelist), time_start,
                   names, path_local)
