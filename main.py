
import numpy as np
import pandas as pd
import scipy.linalg

from struct import unpack
# from Plotting.plot_Bel_coefficients import *
# from SRS_count_coefficients import *
# from Plotting.mean_year import *
from Plotting.video import *
from Plotting.plot_fluxes import *
# from Plotting.plot_eigenvalues import plot_eigenvalues, plot_mean_year
# from Plotting.plot_extreme import *
# from extreme_evolution import *
# from ABCF_coeff_counting import *
from eigenvalues import *
from data_processing import load_ABCFE, load_prepare_fluxes

# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'D:/Nastya/Data/OceanFull/'
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
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    start_year = 2019
    end_year = 2025
    bins_amount = 1000
    days_delta = days_delta5 + days_delta7
    current_shift = 0

    # # normalizing and collecting to bins
    # if not os.path.exists(files_path_prefix + 'Scaling_df.xlsx'):
    #     df = pd.DataFrame(columns=['name', 'start_year', 'min', 'max'])
    # else:
    #     df = pd.read_excel(files_path_prefix + 'Scaling_df.xlsx')

    # sst_grouped = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # sst_min = np.nanmin(sst_grouped)
    # sst_max = np.nanmax(sst_grouped)
    # print(f'SST min = {sst_min}, max = {sst_max}')
    # # df.loc[len(df)] = ['sst', start_year, sst_min, sst_max]
    # sst = (sst_grouped - sst_min)/(sst_max - sst_min)
    # del sst_grouped
    # sst, _ = scale_to_bins(sst, bins_amount)
    # np.save(files_path_prefix + f'SST/SST_{start_year}-{end_year}_norm_scaled.npy', sst)
    # del sst
    # # df.to_excel(files_path_prefix + 'Scaling_df.xlsx')
    #
    # press_grouped = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # press_min = np.nanmin(press_grouped)
    # press_max = np.nanmax(press_grouped)
    # print(f'PRESS min = {press_min}, max = {press_max}')
    # # df.loc[len(df)] = ['press', start_year, press_min, press_max]
    # press = (press_grouped - press_min)/(press_max - press_min)
    # del press_grouped
    # press, _ = scale_to_bins(press, bins_amount)
    # np.save(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_norm_scaled.npy', press)
    # del press
    # # df.to_excel(files_path_prefix + 'Scaling_df.xlsx', index=False)
    #
    # flux_grouped = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # flux_min = np.nanmin(flux_grouped)
    # flux_max = np.nanmax(flux_grouped)
    # print(f'FLUX min = {flux_min}, max = {flux_max}')
    # # df.loc[len(df)] = ['flux', start_year, flux_min, flux_max]
    # flux = (flux_grouped - flux_min)/(flux_max - flux_min)
    # del flux_grouped
    # flux, _ = scale_to_bins(flux, bins_amount)
    # np.save(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_norm_scaled.npy', flux)
    # del flux
    # df.to_excel(files_path_prefix + 'Scaling_df.xlsx', index=False)
    # # ----------------------------------------------------------------------------------------------
    # # count ABF coefficients 3d
    # start_year = 2019
    # end_year = 2025
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #
    # flux = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_norm_scaled.npy')
    # sst = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_norm_scaled.npy')
    # press = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_norm_scaled.npy')
    # count_abfe_coefficients(files_path_prefix,
    #                        mask,
    #                        sst,
    #                        press,
    #                        time_start=0,
    #                        time_end=sst.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='sst-press')
    #
    # count_abfe_coefficients(files_path_prefix,
    #                        mask,
    #                        flux,
    #                        sst,
    #                        time_start=0,
    #                        time_end=sst.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='flux-sst')
    #
    # count_abfe_coefficients(files_path_prefix,
    #                        mask,
    #                        flux,
    #                        press,
    #                        time_start=0,
    #                        time_end=flux.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='flux-press')
    # # ----------------------------------------------------------------------------------------------
    # # Plot fluxes
    # sensible_array = np.load(files_path_prefix + 'Fluxes/sensible_grouped_2019-2022.npy')
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array = np.load(files_path_prefix + 'Fluxes/latent_grouped_2019-2022.npy')
    # latent_array[np.logical_not(mask), :] = np.nan
    # offset = (datetime.datetime(2022, 1, 1) - datetime.datetime(2019, 1, 1)).days
    # plot_fluxes(files_path_prefix, sensible_array, latent_array, offset, offset + 100, 1, datetime.datetime(2022, 1, 1))
    # raise ValueError
    # ---------------------------------------------------------------------------------------
    # collect SST and PRESS to 10 years arrays 3d

    # start_year = 2019
    # end_year = 2025
    # # bins_amount = 1000
    # days_delta = days_delta7 + days_delta5

    # current_shift = 0
    # sst_array = np.zeros((height * width, days_delta * 4))
    # press_array = np.zeros_like(sst_array)
    # for year in range(start_year, end_year):
    #     print(year)
    #     sst_year = np.load(files_path_prefix + f'SST/SST_{year}.npy')
    #     print(sst_year.shape)
    #     sst_array[:, current_shift:current_shift + sst_year.shape[1]] = sst_year[:, :min(sst_year.shape[1], days_delta * 4 - current_shift)]
    #
    #     press_year = np.load(files_path_prefix + f'Pressure/PRESS_{year}.npy')
    #     press_array[:, current_shift:current_shift + sst_year.shape[1]] = press_year[:, :min(press_year.shape[1], days_delta * 4 - current_shift)]
    #
    #     current_shift += sst_year.shape[1]
    #     print()
    #
    # np.save(files_path_prefix + f'SST/SST_{start_year}-{end_year}.npy', sst_array)
    # np.save(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}.npy', press_array)

    # current_shift = 0
    # sensible_array = np.zeros((height * width, days_delta * 4))
    # latent_array = np.zeros_like(sensible_array)
    # for year in ['2019-2023', 2023, 2024]:
    #     print(year)
    #     sensible_year = np.load(files_path_prefix + f'Fluxes/SENSIBLE_{year}.npy')
    #     print(sensible_year.shape)
    #     sensible_array[:, current_shift:current_shift + sensible_year.shape[1]] = sensible_year[:, :min(sensible_year.shape[1], days_delta * 4 - current_shift)]
    #
    #     latent_year = np.load(files_path_prefix + f'Fluxes/LATENT_{year}.npy')
    #     latent_array[:, current_shift:current_shift + latent_year.shape[1]] = latent_year[:, :min(latent_year.shape[1], days_delta * 4 - current_shift)]
    #
    #     current_shift += sensible_year.shape[1]
    #     print()
    #
    # np.save(files_path_prefix + f'Fluxes/SENSIBLE_{start_year}-{end_year}.npy', sensible_array)
    # np.save(files_path_prefix + f'Fluxes/LATENT_{start_year}-{end_year}.npy', latent_array)
    #
    # # get sum fluxes
    # sensible_array = np.load(files_path_prefix + f'Fluxes/SENSIBLE_{start_year}-{end_year}.npy')
    # latent_array = np.load(files_path_prefix + f'Fluxes/LATENT_{start_year}-{end_year}.npy')
    #
    # flux_array = sensible_array + latent_array
    # np.save(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}.npy', flux_array)
    #
    # # Grouping by 1 day
    # sst_array, press_array = load_prepare_fluxes(f'SST/SST_{start_year}-{end_year}.npy',
    #                                              f'Pressure/PRESS_{start_year}-{end_year}.npy',
    #                                              files_path_prefix,
    #                                             prepare=False)
    # print(sst_array.shape)
    # np.save(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy', sst_array)
    # np.save(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy', press_array)
    # del sst_array, press_array
    #
    # flux_array, _ = load_prepare_fluxes(f'Fluxes/FLUX_{start_year}-{end_year}.npy',
    #                                     f'Fluxes/FLUX_{start_year}-{end_year}.npy',
    #                                     files_path_prefix,
    #                                     prepare=False)
    # print(flux_array.shape)
    # np.save(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy', flux_array)

    # # normalizing and collecting to bins
    # if not os.path.exists(files_path_prefix + 'Scaling_df.xlsx'):
    #     df = pd.DataFrame(columns=['name', 'start_year', 'min', 'max'])
    # else:
    #     df = pd.read_excel(files_path_prefix + 'Scaling_df.xlsx')

    # sst_grouped = np.load(files_path_prefix + f'Data/SST/SST_{start_year}-{start_year+10}_grouped.npy')
    # sst_min = np.nanmin(sst_grouped)
    # sst_max = np.nanmax(sst_grouped)
    # print(f'SST min = {sst_min}, max = {sst_max}')
    # # df.loc[len(df)] = ['sst', start_year, sst_min, sst_max]
    # sst = (sst_grouped - sst_min)/(sst_max - sst_min)
    # del sst_grouped
    # sst, _ = scale_to_bins(sst, bins_amount)
    # np.save(files_path_prefix + f'Data/SST/SST_{start_year}-{start_year+10}_norm_scaled.npy', sst)
    # del sst
    # # df.to_excel(files_path_prefix + 'Scaling_df.xlsx')
    #
    # press_grouped = np.load(files_path_prefix + f'Data/Pressure/PRESS_{start_year}-{start_year+10}_grouped.npy')
    # press_min = np.nanmin(press_grouped)
    # press_max = np.nanmax(press_grouped)
    # print(f'PRESS min = {press_min}, max = {press_max}')
    # # df.loc[len(df)] = ['press', start_year, press_min, press_max]
    # press = (press_grouped - press_min)/(press_max - press_min)
    # del press_grouped
    # press, _ = scale_to_bins(press, bins_amount)
    # np.save(files_path_prefix + f'Data/Pressure/PRESS_{start_year}-{start_year+10}_norm_scaled.npy', press)
    # del press
    # # df.to_excel(files_path_prefix + 'Scaling_df.xlsx', index=False)
    #
    # flux_grouped = np.load(files_path_prefix + f'Data/Fluxes/FLUX_{start_year}-{start_year+10}_grouped.npy')
    # flux_min = np.nanmin(flux_grouped)
    # flux_max = np.nanmax(flux_grouped)
    # print(f'FLUX min = {flux_min}, max = {flux_max}')
    # # df.loc[len(df)] = ['flux', start_year, flux_min, flux_max]
    # flux = (flux_grouped - flux_min)/(flux_max - flux_min)
    # del flux_grouped
    # flux, _ = scale_to_bins(flux, bins_amount)
    # np.save(files_path_prefix + f'Data/Fluxes/FLUX_{start_year}-{start_year+10}_norm_scaled.npy', flux)
    # del flux
    # df.to_excel(files_path_prefix + 'Scaling_df.xlsx', index=False)
    #-------------------------------------------------------------------------------------
    # count eigenvalues
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2025_grouped.npy')
    SST_array = np.load(files_path_prefix + f'SST/SST_2019-2025_grouped.npy')
    press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2025_grouped.npy')

    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_1979-1989_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_1979-1989_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_1979-1989_grouped.npy')

    # t = 0
    # cpu_amount = 4
    #
    n_bins = 100
    # # offset = 0
    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    #
    # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_flux_{n_bins}.npy', quantiles_flux)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_sst_{n_bins}.npy', quantiles_sst)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_press_{n_bins}.npy', quantiles_press)
    #
    # print('Counting eigen')
    count_eigenvalues_triplets(files_path_prefix,
                               0, flux_array, SST_array, press_array, mask, offset, n_bins)
    for pair_name in ['Flux-Flux', 'Flux-SST', 'Flux-Pressure', 'SST-SST', 'Pressure-Pressure', 'SST-Pressure']:
        create_video(files_path_prefix, f'videos/Eigenvalues/{pair_name}/', f'Lambdas_', f'{pair_name}_eigenvectors', start=offset)

    # t_start = 0
    # t_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5 + days_delta7
    #
    #
    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Pressure', 'Pressure'), ('Flux', 'SST'),
    #               ('Flux', 'Pressure'), ('SST', 'Pressure')]:
    #     plot_mean_year(files_path_prefix, names)
    #     get_trends(files_path_prefix, t_start, t_end, names)
    #     # plot_eigenvalues_extreme(files_path_prefix, t_start, t_end, 7, names)
    #     print(names)
    #     plot_eigenvalues_extreme(files_path_prefix, t_start, t_end, 30, names)
    #     plot_eigenvalues_extreme(files_path_prefix, t_start, t_end, 365, names)
    #
    # start_date = datetime.datetime(1979, 1, 1) + datetime.timedelta(days= days_delta1 + days_delta2 + days_delta3 + days_delta4)
    # plot_flux_sst_press(files_path_prefix, flux_array, SST_array, press_array, 0, flux_array.shape[1], start_date=start_date,
    #                     start_pic_num= days_delta1 + days_delta2 + days_delta3 + days_delta4)
    # print(days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5 + days_delta7)
    # print('$\\lambda_1=$')


    # # normalizing and collecting to bins
    # start_year = 2019
    # end_year = 2025
    # if not os.path.exists(files_path_prefix + 'Scaling_df.xlsx'):
    #     df = pd.DataFrame(columns=['name', 'start_year', 'min', 'max'])
    # else:
    #     df = pd.read_excel(files_path_prefix + 'Scaling_df.xlsx')

    # sst_grouped = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # sst_min = np.nanmin(sst_grouped)
    # sst_max = np.nanmax(sst_grouped)
    # print(f'SST min = {sst_min}, max = {sst_max}')
    # # df.loc[len(df)] = ['sst', start_year, sst_min, sst_max]
    # sst = (sst_grouped - sst_min)/(sst_max - sst_min)
    # del sst_grouped
    # sst, _ = scale_to_bins(sst, bins_amount)
    # np.save(files_path_prefix + f'SST/SST_{start_year}-{end_year}_norm_scaled.npy', sst)
    # del sst
    # # df.to_excel(files_path_prefix + 'Scaling_df.xlsx')

    # press_grouped = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # press_min = np.nanmin(press_grouped)
    # press_max = np.nanmax(press_grouped)
    # print(f'PRESS min = {press_min}, max = {press_max}')
    # # df.loc[len(df)] = ['press', start_year, press_min, press_max]
    # press = (press_grouped - press_min)/(press_max - press_min)
    # del press_grouped
    # press, _ = scale_to_bins(press, bins_amount)
    # np.save(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_norm_scaled.npy', press)
    # del press
    # # df.to_excel(files_path_prefix + 'Scaling_df.xlsx', index=False)

    # flux_grouped = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # flux_min = np.nanmin(flux_grouped)
    # flux_max = np.nanmax(flux_grouped)
    # print(f'FLUX min = {flux_min}, max = {flux_max}')
    # # df.loc[len(df)] = ['flux', start_year, flux_min, flux_max]
    # flux = (flux_grouped - flux_min)/(flux_max - flux_min)
    # del flux_grouped
    # flux, _ = scale_to_bins(flux, bins_amount)
    # np.save(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_norm_scaled.npy', flux)
    # del flux
    # df.to_excel(files_path_prefix + 'Scaling_df.xlsx', index=False)

    # count ABF coefficients 3d
    # start_year = 2019
    # end_year = 2025
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #
    # flux = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_norm_scaled.npy')
    # sst = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_norm_scaled.npy')
    # press = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_norm_scaled.npy')
    # count_abfe_coefficients(files_path_prefix,
    #                        mask,
    #                        sst,
    #                        press,
    #                        time_start=0,
    #                        time_end=sst.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='sst-press')
    #
    # count_abfe_coefficients(files_path_prefix,
    #                        mask,
    #                        flux,
    #                        sst,
    #                        time_start=0,
    #                        time_end=sst.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='flux-sst')
    #
    # count_abfe_coefficients(files_path_prefix,
    #                        mask,
    #                        flux,
    #                        press,
    #                        time_start=0,
    #                        time_end=flux.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='flux-press')


    # count and plot extreme of coefficients 3d
    # pair_name = 'flux-sst'
    # # pair_name = 'flux-press'
    # # pair_name = 'sst-press'
    #
    # mean_days = 30
    # # mean_days = 365
    # time_start = 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    #
    # if pair_name == 'flux-sst':
    #     names = ('Flux', 'SST')
    # elif pair_name == 'flux-press':
    #     names = ('Flux', 'Pressure')
    # else:
    #     names = ('SST', 'Pressure')
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, e_timelist, borders = load_ABCFE(files_path_prefix,
    #                                                                                  time_start,
    #                                                                                  time_end,
    #                                                                                  load_a=True,
    #                                                                                  load_b=True,
    #                                                                                  path_local=f'Coeff_data_3d/{pair_name}')

    # extract_extreme(files_path_prefix, b_timelist, 'b', time_start, time_end, mean_days,local_path_prefix)
    # plot_extreme(files_path_prefix, 'a', time_start, time_end, mean_days, local_path_prefix, names)
    # plot_extreme(files_path_prefix, 'b', time_start, time_end, mean_days, local_path_prefix, names)
    #
    # for coeff_type in ['a', 'b']:
    #     collect_extreme(files_path_prefix, coeff_type, local_path_prefix, mean_days)
    # for coeff_type in ['a', 'b']:
    #     plot_extreme_3d(files_path_prefix, coeff_type, 1, time_end, mean_days, fit_regression=True,
    #                  fit_sinus=False, fit_fourier_flag=False)



    # local_path_prefix = f'{pair_name}/'
    # extract_extreme(files_path_prefix, a_timelist, 'a', time_start, time_end, mean_days, local_path_prefix)
    # t = 0
    #
    # n_bins = 100
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # count_eigenvalues_triplets(files_path_prefix, 0, flux_array, SST_array, press_array, mask, offset, n_bins)

    # # plot 3d
    # plot_flux_sst_press(files_path_prefix, flux_array, SST_array, press_array, 0, flux_array.shape[1] - 1,
    #                     start_date=datetime.datetime(start_year, 1, 1, 0, 0), start_pic_num=offset)

    # pair_name = 'flux-sst'
    # pair_name = 'flux-press'
    # pair_name = 'sst-press'

    # if pair_name == 'flux-sst':
    #     names = ('Flux', 'SST')
    # elif pair_name == 'flux-press':
    #     names = ('Flux', 'Pressure')
    # else:
    #     names = ('SST', 'Pressure')

    # mean_days = 30
    # mean_days = 365
    # time_start = 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5

    # start_year = 2019
    # if start_year == 1979:
    #     offset = 0
    #     time_end = days_delta1
    # elif start_year == 1989:
    #     offset = days_delta1
    #     time_end = days_delta1 + days_delta2
    # elif start_year == 1999:
    #     offset = days_delta1 + days_delta2
    #     time_end = days_delta1 + days_delta2 + days_delta3
    # elif start_year == 2009:
    #     offset = days_delta1 + days_delta2 + days_delta3
    #     time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # else:
    #     offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #     time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta6
    #
    # time_start = offset
    # if start_year == 2019:
    #     end_year = 2025
    # else:
    #     end_year = start_year + 10

    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    #
    # flux_array = np.diff(flux_array, axis=1)
    # SST_array = np.diff(SST_array, axis=1)
    # press_array = np.diff(press_array, axis=1)

    # extract_extreme(files_path_prefix, flux_array, 'raw', time_start, time_end, mean_days, f'Flux/')
    # extract_extreme(files_path_prefix, SST_array, 'raw', time_start, time_end, mean_days, f'SST/')
    # extract_extreme(files_path_prefix, press_array, 'raw', time_start, time_end, mean_days, f'Pressure/')

    # extract_extreme(files_path_prefix, flux_array, 'diff', time_start, time_end, mean_days, f'Flux/')
    # extract_extreme(files_path_prefix, SST_array, 'diff', time_start, time_end, mean_days, f'SST/')
    # extract_extreme(files_path_prefix, press_array, 'diff', time_start, time_end, mean_days, f'Pressure/')

    # for coeff_type in ['raw']:
    #     collect_extreme(files_path_prefix, coeff_type, 'Flux/', mean_days)
    #     collect_extreme(files_path_prefix, coeff_type, 'SST/', mean_days)
    #     collect_extreme(files_path_prefix, coeff_type, 'Pressure/', mean_days)
    # for coeff_type in ['raw']:
    #     plot_extreme_3d(files_path_prefix, coeff_type, 0, days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta6, mean_days,
    #                     fit_regression=True,
    #                  fit_sinus=False, fit_fourier_flag=False)


    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Flux', 'SST'), ('Flux', 'Pressure'), ('Pressure', 'Pressure')]:
    #     # pair_name = f'{names[0]}-{names[1]}'
    #     plot_eigenvalues_extreme(files_path_prefix, 0, 16554, 30, names)

    # time_start = 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Flux', 'SST'), ('Flux', 'Pressure'), ('Pressure', 'Pressure')]:
    #     pair_name = f'{names[0]}-{names[1]}'
    #     a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, e_timelist, borders = load_ABCFE(files_path_prefix,
    #                                                                                      time_start,
    #                                                                                      time_end,
    #                                                                                      load_a=True,
    #                                                                                      # load_b=True,
    #                                                                                      path_local=f'Coeff_data_3d/{pair_name}')
    #     local_path_prefix = f'{pair_name}/'
    #     extract_extreme(files_path_prefix, a_timelist, 'a', time_start, time_end, mean_days, local_path_prefix)
    #     # extract_extreme(files_path_prefix, b_timelist, 'b', time_start, time_end, mean_days, local_path_prefix)
    #     plot_extreme(files_path_prefix, 'a', time_start, time_end, mean_days, local_path_prefix, names)
    #     # plot_extreme(files_path_prefix, 'b', time_start, time_end, mean_days, local_path_prefix, names)

    # plot_extreme_3d(files_path_prefix, 'a', 1, 16071, mean_days,
    #                 fit_regression=True,
    #              fit_sinus=False, fit_fourier_flag=False)

    # flux_array = np.load()
    # SST_array = np.load(files_path_prefix + f'DATA/SST_2024_hourly.npy')
    # press_array = np.load(files_path_prefix + f'DATA/PRESS_2024_hourly.npy')
    # flux_array = np.zeros_like(SST_array)
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # start_year = 2024
    # SST_array = np.moveaxis(SST_array, (0, 1), (1, 0))
    # press_array = np.moveaxis(press_array, (0, 1), (1, 0))
    # flux_array = np.moveaxis(flux_array, (0, 1), (1, 0))
    #
    # plot_flux_sst_press(files_path_prefix, flux_array, SST_array, press_array, 0, flux_array.shape[1],
    #                     start_date=datetime.datetime(start_year, 1, 1, 0, 0),
    #                     start_pic_num=offset)

