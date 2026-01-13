from SRS_count_coefficients import *

# Parameters
files_path_prefix = 'D://Data/OceanFull/'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    borders = [[0, 1000]]
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
    days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2022, 9, 30, 0, 0) - datetime.datetime(2022, 4, 2, 0, 0)).days
    # ---------------------------------------------------------------------------------------
    # # Plot fluxes
    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022[Oct].npy')
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022[Oct].npy')
    # latent_array[np.logical_not(mask), :] = np.nan
    # offset = (datetime.datetime(2022, 9, 15) - datetime.datetime(2019, 1, 1)).days * 4
    # plot_fluxes(files_path_prefix, sensible_array, latent_array, offset, offset + 10, 4, datetime.datetime(2022, 9, 15))
    # ---------------------------------------------------------------------------------------
    # # Grouping fluxes by 1 day
    # sensible_array, latent_array = load_prepare_fluxes('SENSIBLE_2019-2022[Oct].npy',
    #                                                    'LATENT_2019-2022[Oct].npy',
    #                                                    prepare=False)
    #
    # np.save(files_path_prefix + 'sensible_grouped_2019-2022_2.npy', sensible_array)
    # np.save(files_path_prefix + 'latent_grouped_2019-2022_2.npy', latent_array)
    # ---------------------------------------------------------------------------------------
    # # Scaling fluxes and getting quantiles
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022_2.npy')
    # print(sensible_array.shape)
    # sens_scaled, quantiles = scale_to_bins(sensible_array, 1000)
    # np.save(files_path_prefix + 'sensible_grouped_2019-2022(scaled)_2.npy', sens_scaled)
    # np.save(files_path_prefix + 'Quantiles/sensible_2019-2022(quantiles)_2.npy', np.array(quantiles))
    #
    # latent_array = np.load(files_path_prefix + 'latent_grouped_2019-2022_2.npy')
    # latent_scaled, quantiles = scale_to_bins(latent_array, 1000)
    # np.save(files_path_prefix + 'latent_grouped_2019-2022(scaled)_2.npy', latent_scaled)
    # np.save(files_path_prefix + 'Quantiles/latent_2019-2022(quantiles)_2.npy', np.array(quantiles))
    # ----------------------------------------------------------------------------------------------
    # # Count A and B
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022(scaled)_2.npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_2019-2022(scaled)_2.npy')
    # sensible_array = sensible_array.astype(float)
    # # sensible_array = np.diff(sensible_array)
    # latent_array = latent_array.astype(float)
    # # latent_array = np.diff(latent_array)
    # count_abf_coefficients(files_path_prefix, mask, sensible_array[:, days_delta5-1:], latent_array[:, days_delta5-1:],
    #                        time_start=days_delta5-1, time_end=days_delta5 + days_delta6,
    #                            offset=days_delta1 + days_delta2 + days_delta3 + days_delta4)
    # ---------------------------------------------------------------------------------------
    # # Count C and F
    # time_start = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5 + days_delta6
    # mean_width = 7
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_a=True, load_b=True)
    # count_fraction(files_path_prefix, a_timelist, b_timelist, mask, mean_width, start_idx=time_start)
    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, time_start, 14)
    # count_f_separate_coeff(files_path_prefix, a_timelist, b_timelist, time_start, mean_width)
    # ---------------------------------------------------------------------------------------
    # # Plot coefficients
    # time_start = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5 + days_delta6
    # plot_step = 1
    # delta = 0
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end,
    #                                                                                  load_f=True, load_c=True)
    #
    # # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, delta, time_end-time_start, plot_step, start_pic_num=time_start + delta)
    # plot_c_coeff(files_path_prefix, c_timelist, delta, len(c_timelist), 1, time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, time_end - time_start - delta, plot_step,
    #              start_pic_num=time_start + delta)
    # ---------------------------------------------------------------------------------------
    # # Create video
    # create_video(files_path_prefix, files_path_prefix+'videos/A/', 'A_', 'a_2022', 20, 15706)
    # create_video(files_path_prefix, files_path_prefix+'videos/B/', 'B_', 'b_2022', 20, 15706)
    # create_video(files_path_prefix, files_path_prefix + 'videos/C/', 'C_', 'c_2022', 20, 15706)
    # create_video(files_path_prefix, files_path_prefix + 'videos/FN/', 'FN_', 'f_2022', 20, 15706)
    # # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/FS/', 'FS_', 'FS_daily_mean_7', 10)
    # ---------------------------------------------------------------------------------------


    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2021.npy')
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2021.npy')
    # latent_array[np.logical_not(mask), :] = np.nan
    # offset = (datetime.datetime(2021, 7, 1) - datetime.datetime(2019, 1, 1)).days * 4
    # plot_fluxes(files_path_prefix, sensible_array, latent_array, offset, offset + 10, 1, datetime.datetime(2021, 7, 1))

    # ---------------------------------------------------------------------------------------
    # # Counting A and B

    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')
    # sensible_array = sensible_array.astype(float)
    # sensible_array = np.diff(sensible_array)
    # latent_array = latent_array.astype(float)
    # latent_array = np.diff(latent_array)

    # lost = list()
    # for i in range(1, 15796):
    #     if not os.path.exists(files_path_prefix + f'Coeff_data/{i}_A_sens.npy'):
    #         lost.append(i)
    #
    # print(lost)
    # raise ValueError
    # [3653, 7305, 10958, 14610]
    # hole = 14610

    # time1 = '2009-2019'
    # time2 = '2019-2022'
    #
    # days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    # days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    # days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    # days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    # days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    #
    # sensible1 = np.load(files_path_prefix + f'sensible_grouped_{time1}(scaled).npy')
    # sensible2 = np.load(files_path_prefix + f'sensible_grouped_{time2}(scaled).npy')
    # sensible3 = np.zeros((161*181, sensible1.shape[1] + sensible2.shape[1]))
    # sensible3[:, :sensible1.shape[1]] = sensible1
    # sensible3[:, sensible1.shape[1]:] = sensible2
    # del sensible1
    # del sensible2
    #
    # latent1 = np.load(files_path_prefix + f'latent_grouped_{time1}(scaled).npy')
    # latent2 = np.load(files_path_prefix + f'latent_grouped_{time2}(scaled).npy')
    # latent3 = np.zeros((161*181, latent1.shape[1] + latent2.shape[1]))
    # latent3[:, :latent1.shape[1]] = latent1
    # latent3[:, latent1.shape[1]:] = latent2
    # # del latent1
    # del latent2
    #
    # count_abf_coefficients(files_path_prefix, mask, sensible3, latent3, time_start=latent1.shape[1] - 1, time_end=latent1.shape[1]+1,
    #                            offset=days_delta1 + days_delta2 + days_delta3)

    #
    # for border in borders:
    #     start = border[0]
    #     end = border[1]
    #
    #     count_abf_coefficients(files_path_prefix, mask, sensible_array[:, start:end+1], latent_array[:, start:end+1], time_start=0, time_end=end-start,
    #                            offset=start)

    # offset = 14640 / 4 * 1
    # parallel_AB(4, 'SENSIBLE_1989-1999.npy', 'LATENT_1989-1999.npy', offset)

    # ---------------------------------------------------------------------------------------
    # binary_to_array(files_path_prefix, "s79-21", 'SENSIBLE_1979-1989', datetime.datetime(2019, 1, 1), datetime.datetime(2021, 9, 16))
    # binary_to_array(files_path_prefix, "l79-21", 'LATENT_2009-2019', datetime.datetime(2009, 1, 1), datetime.datetime(2019, 1, 1))
    # ---------------------------------------------------------------------------------------
    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)

    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)
    # ---------------------------------------------------------------------------------------
    # # Estimate a-flux
    # mask = np.array(mask, dtype=int)
    # points = plot_typical_points(files_path_prefix, mask)
    #
    # radius = 2
    # month = 1

    # shutil.rmtree(files_path_prefix + f'Func_repr/a-flux-monthly/{month}')
    # os.mkdir(files_path_prefix + f'Func_repr/a-flux-monthly/{month}')
    # estimate_a_flux_by_months(files_path_prefix, month, point, radius)
    # raise ValueError

    # ---------------------------------------------------------------------------------------
    # # 3D hist for fluxes
    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022.npy')
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022.npy')
    #
    # mask = np.array(mask, dtype=int)
    # points = plot_typical_points(files_path_prefix, mask)
    #
    # radius = 5
    # month = 1

    # point = points[2]
    # if os.path.exists(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})'):
    #     shutil.rmtree(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})')
    # os.mkdir(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})')
    # plot_current_bigpoint(files_path_prefix, mask, point, radius)
    # estimate_flux(files_path_prefix, sensible_array, latent_array, month, point, radius)

    # for point in points:
    #     if os.path.exists(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})'):
    #         shutil.rmtree(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})')
    #     os.mkdir(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})')
    #     plot_current_bigpoint(files_path_prefix, mask, point, radius)
    #     estimate_flux(files_path_prefix, sensible_array, latent_array, month, point, radius)

    # time_start = 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # mean_width = 7
    # window = 30
    #
    # plot_step = 1
    # delta = 0

    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022.npy')
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022.npy')

    # sample_x = sensible_array[:, 0:31]
    # sample_x = sample_x[np.logical_not(np.isnan(sample_x))]
    # sample_y = latent_array[:, 0:31]
    # sample_y = sample_y[np.logical_not(np.isnan(sample_y))]
    # draw_3d_hist(files_path_prefix, sample_x, sample_y, time_start, time_end, postfix='1')


    # plot_fluxes(files_path_prefix, sensible_array, latent_array, time_start, time_end, start_date=datetime.datetime(2022, 1, 1, 0, 0))
    # ---------------------------------------------------------------------------------------
    # # Count C and F

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_a=True, load_b=True)
    # count_fraction(files_path_prefix, a_timelist, b_timelist, mask, mean_width, start_idx=time_start)
    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, time_start, 14)
    # count_f_separate_coeff(files_path_prefix, a_timelist, b_timelist, time_start, mean_width)

    # ---------------------------------------------------------------------------------------
    # # Plot coefficients

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_fs=True)
    # plot_f_coeff(files_path_prefix, fs_timelist, borders, 0, time_end-time_start - delta, start_pic_num=time_start + delta, mean_width=mean_width)

    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, delta, time_end-time_start, plot_step, start_pic_num=time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_c=True)
    # plot_c_coeff(files_path_prefix, c_timelist, delta, len(c_timelist), 1, time_start + delta)

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_fs=True)
    # plot_fs_coeff(files_path_prefix, fs_timelist, borders, delta, len(fs_timelist), 1, time_start + delta, mean_width)

    # ---------------------------------------------------------------------------------------
    # # Create video
    # create_video(files_path_prefix, files_path_prefix+'videos/A/', 'A_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/B/', 'B_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/C/', 'C_', 'c_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/FN/', 'FN_', 'fn_daily_mean', 10)
    # # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/FS/', 'FS_', 'FS_daily_mean_7', 10)

    # ---------------------------------------------------------------------------------------
    # # Grouping fluxes by 1 day
    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)


    # sensible_array, latent_array = load_prepare_fluxes('SENSIBLE_2019-2022.npy',
    #                                                    'LATENT_2019-2022.npy',
    #                                                    prepare=False)
    #
    # np.save(files_path_prefix + 'sensible_grouped_2019-2022.npy', sensible_array)
    # np.save(files_path_prefix + 'latent_grouped_2019-2022.npy', latent_array)
    # ---------------------------------------------------------------------------------------
    # # Scaling fluxes and getting quantiles
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    # print(sensible_array.shape)
    # sens_scaled, quantiles = scale_to_bins(sensible_array, 1000)
    # np.save(files_path_prefix + 'sensible_grouped_2019-2022(scaled).npy', sens_scaled)
    # np.save(files_path_prefix + 'Quantiles/sensible_2019-2022(quantiles).npy', np.array(quantiles))
    #
    # latent_array = np.load(files_path_prefix + 'latent_grouped_2019-2022.npy')
    # latent_scaled, quantiles = scale_to_bins(latent_array, 1000)
    # np.save(files_path_prefix + 'latent_grouped_2019-2022(scaled).npy', latent_scaled)
    # np.save(files_path_prefix + 'Quantiles/latent_2019-2022(quantiles).npy', np.array(quantiles))
    # ----------------------------------------------------------------------------------------------
    # # 3D histogram
    # sensible_array = sensible_array.astype(float)
    # latent_array = latent_array.astype(float)
    #
    # for month in range(1, 2):
    #     days_delta1 = (datetime.datetime(1979, month, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    #     time_start = days_delta1
    #     days_delta2 = (datetime.datetime(1979 + month // 12, month % 12 + 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    #     time_end = days_delta2
    #
    #     sample_x = sensible_array[mask == 1, time_start: time_end].ravel()
        #     sample_y = latent_array[mask == 1, time_start:time_end].ravel()
        #
    #     # months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
    #     #                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    #
    #     draw_3d_hist(files_path_prefix, sample_x, sample_y, time_start, time_end, month)
    # ----------------------------------------------------------------------------------------------
    # # Collecting data into 1 file
    # time_start = 0
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    #
    # sensible_all = np.zeros((161*181, time_end-time_start))
    # latent_all = np.zeros((161*181, time_end-time_start))
    # start = 0
    # for years in ['1979-1989', '1989-1999', '1999-2009', '2009-2019', '2019-2022']:
    #     sens_part = np.load(files_path_prefix + f'sensible_grouped_{years}.npy')
    #     sensible_all[:, start:start + sens_part.shape[1]] = sens_part
    #
    #     lat_part = np.load(files_path_prefix + f'latent_grouped_{years}.npy')
    #     latent_all[:, start:start + lat_part.shape[1]] = lat_part
    #     start += sens_part.shape[1]
    #
    # np.save(files_path_prefix + 'sensible_all.npy', sensible_all)
    # np.save(files_path_prefix + 'latent_all.npy', latent_all)
    # ----------------------------------------------------------------------------------------------
    # q_max = 0.0
    # q_min = 10000
    # for time_start in range(1, 15796, 1000):
    #     time_end = time_start + 1000
    #     a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                      min(time_end, 15796), load_a=True)
    #
    #     a_flat = list()
    #     for a in a_timelist:
    #         a_flat += a[0].flat
    #         a_flat += a[1].flat
    #
    #     del a_timelist
    #     q = np.nanquantile(a_flat, 0.97)
    #     print(q)
    #     q_max = max(q_max, q)
    #     q = np.nanquantile(a_flat, 0.03)
    #     print(q)
    #     print()
    #     q_min = min(q_min, q)
    #     del a_flat
    #
    # print(f'Max {q_max}')
    # print(f'Min {q_min}')
    # time_start = 15341
    # time_end = 15705
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=True)
    # mean_days = 1
    # time_start = 15341
    # time_end = 15705
    # coeff_type = 'a'
    # extract_extreme(files_path_prefix, a_timelist, coeff_type, time_start, time_end, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days)
    #
    # coeff_type = 'b'
    # extract_extreme(files_path_prefix, b_timelist, coeff_type, time_start, time_end, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days)

    # mean_days = 365
    # extract_extreme(files_path_prefix, b_timelist, coeff_type, time_start, time_end, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days)

    # sensible_all = np.load(files_path_prefix + 'sensible_all.npy')
    # latent_all = np.load(files_path_prefix + 'latent_all.npy')
    # check_conditions(files_path_prefix, time_start, time_end, sensible_all, latent_all, mask)
    # ----------------------------------------------------------------------------------------------
    # # Extreme coefficients
    # sensible_all = np.load(files_path_prefix + 'sensible_all.npy')
    # latent_all = np.load(files_path_prefix + 'latent_all.npy')
    # extract_extreme_coeff_flux(files_path_prefix, 'a', time_start, time_end, sensible_all, latent_all, 365)
    # plot_extreme_coeff_flux(files_path_prefix, 'a', time_start, time_end, 30)

    # ----------------------------------------------------------------------------------------------
    # # A-B distribution
    # time_start = days_delta1 + days_delta2 + days_delta3
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # mask = np.array(mask, dtype=int)
    # points = plot_typical_points(files_path_prefix, mask)
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=False)
    # for point in points:
    #     plot_estimate_ab_distributions(files_path_prefix, a_timelist, b_timelist,time_start, time_end, point)
    # ----------------------------------------------------------------------------------------------
    # # Mean year
    # mean_years = np.zeros((365, 161, 181))
    # for year in tqdm.tqdm(range(1979, 2022)):
    #     time_start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=1979, month=1, day=1)).days
    #     time_end = time_start + 364
    #     a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                      time_end, load_c=True)
    #     for day in range(363):
    #         mean_years[day, :, :] += c_timelist[day][1]
    #
    # mean_years /= (2022 - 1979)
    # np.save(files_path_prefix + f'Mean_year/C_1.npy', mean_years)

    # plot_mean_year(files_path_prefix, 'C_1')
    # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # # Getting quantiles of coefficients by 10-year interval
    # time_start = days_delta1 + days_delta2 + days_delta3
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # mask = np.array(mask, dtype=int)
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=True)
    # # print(borders[2])
    # # print(borders[3])
    #
    # start_year = 2009
    # b_list = list()
    # for year in tqdm.tqdm(range(10)):
    #     start_date = (datetime.datetime(start_year + year, 1, 1) - datetime.datetime(start_year + year, 1, 1)).days
    #     end_date = (datetime.datetime(start_year + year, 12, 31) - datetime.datetime(start_year + year, 1, 1)).days
    #     for day in range(start_date, end_date):
    #         b_list.append(list(b_timelist[day][2][~np.isnan(b_timelist[day][0])].flatten()))
    #
    # print(scipy.stats.mstats.mquantiles(b_list, [0.999]))
    # # print(sum(i > 1000 for i in b_list)*1.0/len(b_list) * 100)
    # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # # Components method and Belyaev-Korolev comparison
    # points = plot_typical_points(files_path_prefix, mask)
    # n_components = 3
    # radius = 3
    #
    # ticks_by_day = 4
    # step_ticks = 2
    # window_width = ticks_by_day * 1
    #
    # # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022.npy')
    # # sensible_array = sensible_array[:, :365*ticks_by_day]
    # # flux_type = 'sensible'
    # # data_array = sensible_array
    #
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022.npy')
    # latent_array = latent_array[:, :365*ticks_by_day]
    # flux_type = 'latent'
    # data_array = latent_array
    #
    # time_start = 1
    # time_end = data_array.shape[1]
    # timedelta = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #
    # for point in tqdm.tqdm([points[0]]):
    #     point_size = (radius * 2 + 1) ** 2
    #     point_bigger = [point]
    #     for i in range(-radius, radius + 1):
    #         for j in range(-radius, radius + 1):
    #             if mask[i*181 + j]:
    #                 point_bigger.append((point[0] + i, point[1]+j))
    #             else:
    #                 point_size -= 1
    #
    #     sample = np.zeros((point_size, time_end - time_start - 1))
    #     for i in range(point_size):
    #         p = point_bigger[i]
    #         sample[i, :] = np.diff(data_array[p[0]*181 + p[1], time_start:time_end])
    #
    #     # reshape
    #     sample = sample.transpose().flatten()
    #
    #     # apply EM
    #     point_df = hybrid(sample, window_width * point_size, n_components, EM_steps=1, step=step_ticks*point_size)
    #     if not os.path.exists(files_path_prefix + f'Components/{flux_type}/raw'):
    #         os.mkdir(files_path_prefix + f'Components/{flux_type}/raw')
    #     point_df.to_excel(files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx', index=False)
    #     df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx')
    #     new_df, new_n_components = cluster_components(df, n_components, point, files_path_prefix, flux_type, True)
    #     if not os.path.exists(files_path_prefix + f'Components/{flux_type}/components-xlsx'):
    #         os.mkdir(files_path_prefix + f'Components/{flux_type}/components-xlsx')
    #     new_df.to_excel(files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx', index=False)
    #     plot_components(new_df, new_n_components, point, files_path_prefix, flux_type)
    #     plot_a_sigma(df, n_components, point, files_path_prefix, flux_type)
    #
    #
    #     count_Bel_Kor_difference(files_path_prefix,
    #                              time_start,
    #                              time_end//ticks_by_day,
    #                              point_bigger,
    #                              point_size,
    #                              point,
    #                              n_components,
    #                              window_width,
    #                              ticks_by_day,
    #                              step_ticks,
    #                              timedelta,
    #                              flux_type)
    #
    #     plot_difference_1d(files_path_prefix,
    #                        time_start,
    #                        time_end//ticks_by_day,
    #                        point,
    #                        window_width,
    #                        radius,
    #                        ticks_by_day,
    #                        step_ticks,
    #                        flux_type)
    # ----------------------------------------------------------------------------------------------
    # # Creating synthetic flux and counting Bel and Kor methods for it and plotting the difference
    # create_synthetic_data_1d(files_path_prefix, time_start=0, time_end=100)
    # flux = np.load(f'{files_path_prefix}/Synthetic/flux_full.npy')
    # a_array = np.load(f'{files_path_prefix}/Synthetic/A_full.npy')
    # b_array = np.load(f'{files_path_prefix}/Synthetic/B_full.npy')
    # plot_synthetic_flux(files_path_prefix, flux, 0, 5, a_array, b_array)
    # raise ValueError
    # count_synthetic_Bel(files_path_prefix, 1, flux, None, None, time_start=0, time_end=100, quantiles=50)
    # count_synthetic_Korolev(files_path_prefix, flux, 0, 100, 50, 3, synthetic=True)
    # count_optimization(files_path_prefix, flux, 0, 100, 3, 50)

    # for point in [(0, j) for j in range(30)]:
    #     # count_Bel_Kor_difference(files_path_prefix, 1, 100, point, '')
    #     plot_difference_1d_synthetic(files_path_prefix, point, 3, 1, 99, 'A')
    #     plot_difference_1d_synthetic(files_path_prefix, point, 3, 1, 99, 'B')
    # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    # sensible_array = sensible_array.transpose()
    # sensible_array = sensible_array.reshape((sensible_array.shape[0], 161, 181))
    # flux_type = 'sensible'

    # latent_array = np.load(files_path_prefix + 'latent_grouped_2019-2022.npy')
    # latent_array = latent_array.transpose()
    # latent_array = latent_array.reshape((latent_array.shape[0], 161, 181))
    # flux_type = 'latent'
    #
    # print('Counting Bel')
    # start_time = time.time()
    # count_1d_Bel(files_path_prefix,
    #              latent_array,
    #              time_start=0,
    #              time_end=len(latent_array),
    #              path=f'Components/{flux_type}/',
    #              quantiles_amount=1000,
    #              start_index=days_delta1+days_delta2+days_delta3 + days_delta4)
    #
    # print("--- Bel %s seconds ---" % (time.time() - start_time))
    # print(f'{(time.time() - start_time) / len(latent_array):.5f} seconds per iteration')

    # print('Counting Kor')
    # start_time = time.time()
    # count_1d_Korolev(files_path_prefix,
    #                  latent_array,
    #                  time_start=0,
    #                  time_end=len(latent_array),
    #                  path=f'Components/{flux_type}/',
    #                  quantiles_amount=50,
    #                  n_components=3,
    #                  start_index=days_delta1+days_delta2+days_delta3+days_delta4)
    # print("--- Kor %s seconds ---" % (time.time() - start_time))
    # print(f'{(time.time() - start_time) / len(latent_array):.5f} seconds per iteration')

    # plot_methods_compare(files_path_prefix, 0, 100, sensible_array, 'sensible', 'A', 161, 181)
    # plot_methods_compare(files_path_prefix, 0, 100, sensible_array, 'sensible', 'B', 161, 181)
    # points = plot_typical_points(files_path_prefix, mask)
    # for point in points:
    #     count_Bel_Kor_difference(files_path_prefix, 1, 100, point, 'sensible')
    #     plot_difference_1d(files_path_prefix, point, 3, 1, 99, 'A', 'sensible')
    #     plot_difference_1d(files_path_prefix, point, 3, 1, 99, 'B', 'sensible')

    # coeff_type = 'B'
    # # method = 'Bel'
    # # # count_mean_year(files_path_prefix, f'Components/{flux_type}/{method}/daily/', coeff_type=coeff_type,
    # # #                 postfix=f'{flux_type}_{method}', mask=mask.reshape((161, 181)))
    # # method = 'Kor'
    # # count_mean_year(files_path_prefix, f'Components/{flux_type}/{method}/daily/', coeff_type=coeff_type,
    # #                 postfix=f'{flux_type}_{method}', mask=mask.reshape((161, 181)))
    # mean_year = np.load(files_path_prefix + f'Mean_year/{coeff_type}_2009-2019_{flux_type}_Bel.npy')
    # Bel_min = np.nanmin(mean_year)
    # Bel_max = np.nanmax(mean_year)
    # method = 'Bel'
    # plot_mean_year_1d(files_path_prefix, coeff_type=coeff_type, postfix=f'{flux_type}_{method}', coeff_min=Bel_min, coeff_max=Bel_max)
    # method = 'Kor'
    # plot_mean_year_1d(files_path_prefix, coeff_type=coeff_type, postfix=f'{flux_type}_{method}', coeff_min=Bel_min, coeff_max=Bel_max)
    #

    # ---------------------------------------------------------------------------------------
    # # Plot Bel coefficients
    # time_start = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # plot_step = 1
    # delta = 0
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_a=True, load_b=True,
    #                                                                                  load_f=True, load_c=True)
    #
    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, delta, time_end-time_start, plot_step, start_pic_num=time_start + delta)
    # plot_c_coeff(files_path_prefix, c_timelist, delta, len(c_timelist), 1, time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, time_end - time_start - delta, plot_step,
    #              start_pic_num=time_start + delta)
    # ---------------------------------------------------------------------------------------
    # # Plot fluxes
    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022[Oct].npy')
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022[Oct].npy')
    # latent_array[np.logical_not(mask), :] = np.nan
    # offset = (datetime.datetime(2022, 1, 1) - datetime.datetime(2019, 1, 1)).days * 4
    # plot_fluxes(files_path_prefix, sensible_array, latent_array, offset, offset + 100, 4, datetime.datetime(2022, 1, 1))

    # ---------------------------------------------------------------------------------------
    # # count and plot fluxes correlations
    # sensible_array = np.load('E://Nastya/Data/OceanFull/' + 'sensible_grouped_1989-1999.npy')
    # sensible_array[np.logical_not(mask), :] = np.nan
    # sensible_array = sensible_array.reshape((sensible_array.shape[1], 161, 181))
    #
    # latent_array = np.load('E://Nastya/Data/OceanFull/' + 'latent_grouped_1989-1999.npy')
    # latent_array[np.logical_not(mask), :] = np.nan
    # latent_array = latent_array.reshape((latent_array.shape[1], 161, 181))
    #
    # count_correlations(files_path_prefix, sensible_array, latent_array, days_delta1, 14, 1)
    # ---------------------------------------------------------------------------------------
    # # Plot differences of mean years
    # coeff_type = 'B'
    # flux_type = 'sensible'
    # start_year = 2009
    # end_year = 2019
    # postfix = 'absolute_difference'
    # mean_year_Bel = np.load(files_path_prefix + f'Mean_year/{coeff_type}_2009-2019_{flux_type}_Bel.npy')
    # mean_year_Kor = np.load(files_path_prefix + f'Mean_year/{coeff_type}_2009-2019_{flux_type}_Kor.npy')
    # mean_year = np.abs(mean_year_Kor - mean_year_Bel)
    # coeff_min = np.nanmin(mean_year)
    # coeff_max = np.nanmax(mean_year)
    #
    # fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # # fig.suptitle(f'{coeff_type} absolute differences {flux_type} mean year', fontsize=30)
    # axs[0][0].title.set_text('February, 15')
    # axs[0][1].title.set_text('April, 15')
    # axs[0][2].title.set_text('June, 15')
    # axs[1][0].title.set_text('August, 15')
    # axs[1][1].title.set_text('October, 15')
    # axs[1][2].title.set_text('December, 15')
    # img = [None for _ in range(6)]
    # cax = [None for _ in range(6)]
    # days = [(datetime.datetime(start_year, i * 2, 15) - datetime.datetime(start_year, 1, 2)).days for i in range(1, 7)]
    #
    # # cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    # cmap = plt.get_cmap('Reds')
    # cmap.set_bad('darkgreen', 1.0)
    # for i in range(6):
    #     divider = make_axes_locatable(axs[i // 3][i % 3])
    #     cax[i] = divider.append_axes('right', size='5%', pad=0.3)
    #     img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
    #                                        interpolation='none',
    #                                        cmap=cmap,
    #                                        vmin=coeff_min,
    #                                        vmax=coeff_max)
    #
    #     x_label_list = ['90W', '60W', '30W', '0']
    #     y_label_list = ['EQ', '30N', '60N', '80N']
    #     xticks = [0, 60, 120, 180]
    #     yticks = [160, 100, 40, 0]
    #
    #     axs[i // 3][i % 3].set_xticks(xticks)
    #     axs[i // 3][i % 3].set_yticks(yticks)
    #     axs[i // 3][i % 3].set_xticklabels(x_label_list)
    #     axs[i // 3][i % 3].set_yticklabels(y_label_list)
    #     fig.colorbar(img[i], cax=cax[i], orientation='vertical')
    #
    # plt.tight_layout()
    # fig.savefig(files_path_prefix + f'videos/Mean_year/{coeff_type}_{flux_type}_{start_year}-{end_year}_{postfix}.png')
    # ----------------------------------------------------------------------------------------------
    # collect SST and PRESS to 10 years arrays 3d

    # start_year = 2019
    # end_year = 2025
    # bins_amount = 1000
    # days_delta = days_delta5 + days_delta6
    # current_shift = 0
    # sst_array = np.zeros((height * width, days_delta * 4))
    # press_array = np.zeros_like(sst_array)
    # for year in range(start_year, end_year):
    #     print(year)
    #     sst_year = np.load(files_path_prefix + f'SST/SST_{year}.npy')
    #     print(sst_year.shape)
    #     sst_array[:, current_shift:current_shift + sst_year.shape[1]] = sst_year
    #
    #     press_year = np.load(files_path_prefix + f'Pressure/PRESS_{year}.npy')
    #     press_array[:, current_shift:current_shift + sst_year.shape[1]] = press_year
    #
    #     current_shift += sst_year.shape[1]
    #     print()
    #
    # np.save(files_path_prefix + f'SST/SST_{start_year}-{end_year}.npy', sst_array)
    # np.save(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}.npy', press_array)

    # sensible_array = np.zeros((height * width, days_delta * 4))
    # latent_array = np.zeros_like(sensible_array)
    # for year in ['2019-2023', 2023, 2024]:
    #     print(year)
    #     sensible_year = np.load(files_path_prefix + f'Fluxes/SENSIBLE_{year}.npy')
    #     print(sensible_year.shape)
    #     sensible_array[:, current_shift:current_shift + sensible_year.shape[1]] = sensible_year
    #
    #     latent_year = np.load(files_path_prefix + f'Fluxes/LATENT_{year}.npy')
    #     latent_array[:, current_shift:current_shift + latent_year.shape[1]] = latent_year
    #
    #     current_shift += sensible_year.shape[1]
    #     print()
    #
    # np.save(files_path_prefix + f'Fluxes/SENSIBLE_{start_year}-{end_year}.npy', sensible_array)
    # np.save(files_path_prefix + f'Fluxes/LATENT_{start_year}-{end_year}.npy', latent_array)

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

    # ----------------------------------------------------------------------------------------------
    # # plot 3d
    # flux = np.load(files_path_prefix + f'flux_{year_str}_grouped.npy')
    # sst = np.load(files_path_prefix + f'sst_{year_str}_grouped.npy')
    # press = np.load(files_path_prefix + f'press_{year_str}_grouped.npy')
    # plot_flux_sst_press(files_path_prefix, flux, sst, press, 0, flux.shape[1] - 1,
    #                     start_date=datetime.datetime(int(year_str), 1, 1, 0, 0), start_pic_num=days_delta7)

    # ----------------------------------------------------------------------------------------------
    # count ABF coefficients 3d
    # start_year = 2019
    # end_year = 2023
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #
    # flux = np.load(files_path_prefix + f'Data/Fluxes/FLUX_{start_year}-{end_year}_norm_scaled.npy')
    # sst = np.load(files_path_prefix + f'Data/SST/SST_{start_year}-{end_year}_norm_scaled.npy')
    # press = np.load(files_path_prefix + f'Data/Pressure/PRESS_{start_year}-{end_year}_norm_scaled.npy')
    # count_abf_coefficients(files_path_prefix,
    #                        mask,
    #                        sst,
    #                        press,
    #                        time_start=0,
    #                        time_end=sst.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='sst-press')
    #
    # count_abf_coefficients(files_path_prefix,
    #                        mask,
    #                        flux,
    #                        sst,
    #                        time_start=0,
    #                        time_end=sst.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='flux-sst')
    #
    # count_abf_coefficients(files_path_prefix,
    #                        mask,
    #                        flux,
    #                        press,
    #                        time_start=0,
    #                        time_end=flux.shape[1] - 1,
    #                        offset=offset,
    #                        pair_name='flux-press')
    # ----------------------------------------------------------------------------------------------
    # # count C and FS 3d
    # time_start = days_delta1 + days_delta2 + days_delta3 + days_delta4 + 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # offset = 0
    # # pair_name = 'flux-sst'
    # # pair_name = 'flux-press'
    # pair_name = 'sst-press'
    # if pair_name == 'flux-sst':
    #     names = ('Flux', 'SST')
    # elif pair_name == 'flux-press':
    #     names = ('Flux', 'Pressure')
    # else:
    #     names = ('SST', 'Pressure')
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix,
    #                                                                                  time_start,
    #                                                                                  time_end,
    #                                                                                  load_a=True,
    #                                                                                  load_b=True,
    #                                                                                  path_local=f'Coeff_data_3d/{pair_name}'
    #                                                                                  )
    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, start_idx=time_start+offset, pair_name=pair_name)
    # count_f_separate_coeff(files_path_prefix, a_timelist, b_timelist, start_idx=time_start+offset, mean_width=7,
    #                        pair_name=pair_name)
    # ----------------------------------------------------------------------------------------------
    # # plot coefficients 3d
    # # pair_name = 'flux-sst'
    # pair_name = 'flux-press'
    # # pair_name = 'sst-press'
    #
    # time_start = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # offset = 0
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix,
    #                                                                                  time_start + offset + 1,
    #                                                                                  time_end,
    #                                                                                  load_a=True,
    #                                                                                  load_b=True,
    #                                                                                  path_local=f'Coeff_data_3d/{pair_name}'
    #                                                                                  )
    #
    # if pair_name == 'flux-sst':
    #     names = ('Flux', 'SST')
    # elif pair_name == 'flux-press':
    #     names = ('Flux', 'Pressure')
    # else:
    #     names = ('SST', 'Pressure')
    #
    # plot_ab_coefficients(files_path_prefix,
    #                      a_timelist,
    #                      b_timelist,
    #                      borders,
    #                      0,
    #                      len(a_timelist),
    #                      start_pic_num=time_start + offset,
    #                      names=names,
    #                      path_local=f'3D/{pair_name}/',
    #                      start_date=datetime.datetime(1979, 1, 1, 0, 0))

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix,
    #                                                                                  time_start + offset + 1,
    #                                                                                  time_end,
    #                                                                                  load_c=True,
    #                                                                                  path_local=f'Coeff_data_3d/{pair_name}'
    #                                                                                  )
    # plot_c_coeff(files_path_prefix, c_timelist, 0, len(c_timelist), start_pic_num=time_start + offset,
    #              pair_name=pair_name, path_local=f'3D/{pair_name}/')
    # del c_timelist
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix,
    #                                                                                  time_start + offset + 1,
    #                                                                                  time_end - 14,
    #                                                                                  load_fs=True,
    #                                                                                  path_local=f'Coeff_data_3d/{pair_name}'
    #                                                                                  )
    # plot_fs_coeff(files_path_prefix, fs_timelist, borders, 0, len(fs_timelist), start_pic_num=time_start + offset,
    #               names=names, path_local=f'3D/{pair_name}/')
    # ----------------------------------------------------------------------------------------------
    # count correlations 3d
    # start_year = 1979
    # offset = 0

    # flux = np.load(files_path_prefix + f'Data/Fluxes/FLUX_{start_year}-{start_year+10}_grouped.npy')
    # sst = np.load(files_path_prefix + f'Data/SST/SST_{start_year}-{start_year+10}_grouped.npy')
    # press = np.load(files_path_prefix + f'Data/Pressure/PRESS_{start_year}-{start_year+10}_grouped.npy')
    #
    # count_correlations(files_path_prefix, flux, sst, offset, observations_per_day=1, names=('Flux', 'SST'))
    # count_correlations(files_path_prefix, flux, press, offset, observations_per_day=1, names=('Flux', 'Pressure'))
    # count_correlations(files_path_prefix, sst, press, offset, observations_per_day=1, names=('SST', 'Pressure'))
    # ----------------------------------------------------------------------------------------------
    # # Count Korolev AB coefficients
    # start_year = 2009
    # offset = days_delta1 + days_delta2 + days_delta3
    # flux_type = 'sensible'
    #
    # # Count Korolev
    # array = np.load(files_path_prefix + f'Data/{flux_type}/{flux_type}_grouped_{start_year}-{start_year+10}.npy')
    # array = array.transpose()
    # array = array.reshape((array.shape[0], 161, 181))
    #
    # count_1d_Korolev(files_path_prefix,
    #                  array,
    #                  time_start=0,
    #                  time_end=len(array),
    #                  path=f'Components/{flux_type}/',
    #                  quantiles_amount=50,
    #                  n_components=3,
    #                  start_index=offset)
    #
    # del array
    # ----------------------------------------------------------------------------------------------
    # count and plot mean year for 10-years interval
    # method = 'Kor'
    # for start_year in [1979, 1989, 1999, 2009, 2019]:
    #     if start_year == 2019:
    #         end_year = 2022
    #     else:
    #         end_year = start_year + 10
    #     for flux_type in ['sensible', 'latent']:
    #         for coeff_type in ['A', 'B']:
                # count_mean_year(files_path_prefix, coeff_type=coeff_type,
                #                 method=method, mask=mask.reshape((161, 181)), start_year=start_year,
                #                 end_year=end_year, flux_type=flux_type)
                # plot_mean_year_1d(files_path_prefix, coeff_type=coeff_type, flux_type=flux_type, method=method,
                #                   start_year=start_year, end_year=end_year)

                # mean_year_Bel = np.load(files_path_prefix + f'Mean_year/{coeff_type}_{start_year}-{end_year}_{flux_type}_Bel.npy')
                # mean_year_Kor = np.load(files_path_prefix + f'Mean_year/{coeff_type}_{start_year}-{end_year}_{flux_type}_Kor.npy')
                # mean_year = np.abs(mean_year_Kor - mean_year_Bel)
                # plot_mean_year_1d_difference(files_path_prefix, mean_year, start_year, end_year, coeff_type, flux_type)
    # ----------------------------------------------------------------------------------------------
    # # count and plot mean year for 43-years interval
    # method = 'Bel'
    # start_year = 1979
    # end_year = 2022
    # for flux_type in ['sensible', 'latent']:
    #     for coeff_type in ['A', 'B']:
    #         # count_mean_year(files_path_prefix, coeff_type=coeff_type,
    #         #                 method=method, mask=mask.reshape((161, 181)), start_year=start_year,
    #         #                 end_year=end_year, flux_type=flux_type)
    #         # plot_mean_year_1d(files_path_prefix, coeff_type=coeff_type, flux_type=flux_type, method=method,
    #         #                   start_year=start_year, end_year=end_year)
    #
    #         mean_year_Bel = np.load(files_path_prefix + f'Mean_year/{coeff_type}_{start_year}-{end_year}_{flux_type}_Bel.npy')
    #         mean_year_Kor = np.load(files_path_prefix + f'Mean_year/{coeff_type}_{start_year}-{end_year}_{flux_type}_Kor.npy')
    #         mean_year = np.abs(mean_year_Kor - mean_year_Bel)
    #         plot_mean_year_1d_difference(files_path_prefix, mean_year, start_year, end_year, coeff_type, flux_type)
    # ----------------------------------------------------------------------------------------------
    # # get sum fluxes
    # days_delta6 = (datetime.datetime(2022, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # sensible1 = np.load(files_path_prefix + f'Fluxes/SENSIBLE_2019-2022.npy')
    # latent1 = np.load(files_path_prefix + f'Fluxes/LATENT_2019-2022.npy')
    #
    # sensible2 = np.load(files_path_prefix + f'Fluxes/SENSIBLE_2022.npy')
    # latent2 = np.load(files_path_prefix + f'Fluxes/LATENT_2022.npy')
    # print(sensible2.shape) # =4*365
    #
    # sensible_all = np.zeros((sensible1.shape[0], 4* days_delta5))
    # print(sensible_all.shape) #= 4*1461
    #
    # print(sensible_all.shape[1] - days_delta6*4)
    # latent_all = np.zeros_like(sensible_all)
    # sensible_all[:, :sensible1.shape[1]] = sensible1
    # latent_all[:, :sensible1.shape[1]] = latent1
    #
    # sensible_all[:, -sensible2.shape[1]:] = sensible2
    # latent_all[:, -sensible2.shape[1]:] = latent2
    #
    # np.save(files_path_prefix + f'Fluxes/SENSIBLE_2019-2023.npy', sensible_all)
    # np.save(files_path_prefix + f'Fluxes/LATENT_2019-2023.npy', latent_all)
    #
    # flux_array = sensible_all + latent_all
    # np.save(files_path_prefix + f'Fluxes/FLUX_2019-2023.npy', flux_array)
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

    # mean_days = 365
    # time_start = 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    #
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

    # for coeff_type in ['a', 'b']:
    #     collect_extreme(files_path_prefix, coeff_type, local_path_prefix, mean_days)
    # for coeff_type in ['a', 'b']:
    #     plot_extreme_3d(files_path_prefix, coeff_type, 1, 16071, mean_days, fit_regression=True,
    #                  fit_sinus=True, fit_fourier_flag=True)
    # ----------------------------------------------------------------------------------------------
    # # eigenvalues
    # print(f'start year {start_year}')
    # end_year = start_year + 10
    #
    # if start_year == 1979:
    #     offset = 0
    # elif start_year == 1989:
    #     offset = days_delta1
    # elif start_year == 1999:
    #     offset = days_delta1 + days_delta2
    # elif start_year == 2009:
    #     offset = days_delta1 + days_delta2 + days_delta3
    # else:
    #     offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #     end_year = 2025
    #
    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # t = t_start
    #
    # n_bins = 100

    # count_eigenvalues_triplets(files_path_prefix, 0, flux_array, SST_array, press_array, mask, offset, n_bins)

    # print('Counting mean years', flush=True)
    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Flux', 'SST'), ('Flux', 'Pressure')]:
    # for names in [('Pressure', 'Pressure')]:
    #     count_mean_year(files_path_prefix, 1979, 2025, names, mask)

    # print('Getting trends', flush=True)
    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Flux', 'SST'), ('Flux', 'Pressure')]:
    # for names in [('Pressure', 'Pressure')]:
    #     get_trends(files_path_prefix, 0, days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta6, names)

    # for names in [('Pressure', 'Pressure')]:
    #     pair_name = f'{names[0]}-{names[1]}'
    #     create_video(files_path_prefix, f'videos/Eigenvalues/{pair_name}/', f'Lambdas_', f'{pair_name}_eigenvalues',
    #                  start=14610)

    # ----------------------------------------------------------------------------------------------
    # collect SST and PRESS to 10 years arrays 3d

    # start_year = 2019
    # end_year = 2025
    # bins_amount = 1000
    # days_delta = days_delta6

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
    # start_year = 2019
    # end_year = 2026
    # bins_amount = 1000
    # days_delta = days_delta5 + days_delta6
    # current_shift = 0
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
    # end_year = 2026
    # bins_amount = 1000
    # days_delta = days_delta5 + days_delta6

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
    # for year in ['2019-2023', 2023, 2024, 2025]:
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
    # raise ValueError
    # # get sum fluxes
    # sensible_array = np.load(files_path_prefix + f'Fluxes/SENSIBLE_{start_year}-{end_year}.npy')
    # latent_array = np.load(files_path_prefix + f'Fluxes/LATENT_{start_year}-{end_year}.npy')
    #
    # flux_array = sensible_array + latent_array
    # np.save(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}.npy', flux_array)


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

    # sensible_array, latent_array = load_prepare_fluxes(f'Fluxes/SENSIBLE_{start_year}-{end_year}.npy',
    #                                     f'Fluxes/LATENT_{start_year}-{end_year}.npy',
    #                                     files_path_prefix,
    #                                     prepare=False)
    # print(sensible_array.shape)
    # np.save(files_path_prefix + f'Fluxes/SENSIBLE_{start_year}-{end_year}_grouped.npy', sensible_array)
    # np.save(files_path_prefix + f'Fluxes/LATENT_{start_year}-{end_year}_grouped.npy', latent_array)
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
    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2025_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_2019-2025_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2025_grouped.npy')

    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_1979-1989_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_1979-1989_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_1979-1989_grouped.npy')

    # t = 0
    # cpu_amount = 4
    #
    # n_bins = 100
    # # # offset = 0
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    #
    # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_flux_{n_bins}.npy', quantiles_flux)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_sst_{n_bins}.npy', quantiles_sst)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_press_{n_bins}.npy', quantiles_press)
    #
    # print('Counting eigen')
    # print(offset  + days_delta6)
    # # raise ValueError
    # count_eigenvalues_triplets(files_path_prefix,
    #                            0, flux_array, SST_array, press_array, mask,
    #                            offset, n_bins)
    # for pair_name in ['Flux-Flux', 'Flux-SST', 'Flux-Pressure', 'SST-SST', 'Pressure-Pressure', 'SST-Pressure']:
    #     create_video(files_path_prefix, f'videos/Eigenvalues/{pair_name}/', f'Lambdas_', f'{pair_name}_eigenvectors', start=offset)

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

    # create_synthetic_data_2d(files_path_prefix, 100, 100, 0, 1000)

    # # ----------------------------------------------------------------------------------------------
    # # Creating synthetic flux and counting Bel and Kor methods for it and plotting the difference
    # # create_synthetic_data_1d(files_path_prefix, time_start=0, time_end=100)
    # flux = np.load(f'{files_path_prefix}Synthetic/flux_full.npy')
    # a_array = np.load(f'{files_path_prefix}Synthetic/A_full.npy')
    # b_array = np.load(f'{files_path_prefix}Synthetic/B_full.npy')
    # # # plot_synthetic_flux(files_path_prefix, flux, 0, 5, a_array, b_array)
    #
    # quantiles = np.array([200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500])
    # rmse_a_bel = np.zeros(len(quantiles), dtype=float)
    # rmse_b_bel = np.zeros(len(quantiles), dtype=float)
    # rmse_a_kor = np.zeros(len(quantiles), dtype=float)
    # rmse_b_kor = np.zeros(len(quantiles), dtype=float)
    # for q in range(len(quantiles)):
    #     quantiles_amount = quantiles[q]
    #     count_1d_Bel(files_path_prefix, flux, 0, 100, 'Synthetic/', quantiles_amount)
    #     count_1d_Korolev(files_path_prefix, flux, 0, 100, 'Synthetic/', quantiles_amount,)
    #
    #     rmse_Bel = [0.0, 0.0]
    #     rmse_Kor = [0.0, 0.0]
    #     points_j_amount = 10
    #     points_i_amount = 10
    #     for point in tqdm.tqdm([(i, j) for i in range(points_i_amount) for j in range(points_j_amount)]):
    #         collect_point(files_path_prefix, 1, 100, point, 'Synthetic/', 'Bel')
    #         collect_point(files_path_prefix, 1, 100, point, 'Synthetic/', 'Kor')
    #         # count_Bel_Kor_difference(files_path_prefix, 1, 100, point, '')
    #         # plot_difference_1d_synthetic(files_path_prefix, point, 3, 1, 99, 'A')
    #         # plot_difference_1d_synthetic(files_path_prefix, point, 3, 1, 99, 'B')
    #
    #         a_Bel = np.load(files_path_prefix + 'Synthetic/' + f'Bel/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Bel = np.load(files_path_prefix + 'Synthetic/' + f'Bel/points/point_({point[0]}, {point[1]})-B.npy')
    #         a_Kor = np.load(files_path_prefix + 'Synthetic/' + f'Kor/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Kor = np.load(files_path_prefix + 'Synthetic/' + f'Kor/points/point_({point[0]}, {point[1]})-B.npy')
    #         # count rmse
    #         rmse_Bel[0] += math.sqrt(sum((a_Bel - a_array[:, point[0], point[1]]) ** 2))
    #         rmse_Bel[1] += math.sqrt(sum((b_Bel - b_array[:, point[0], point[1]]) ** 2))
    #         rmse_Kor[0] += math.sqrt(sum((a_Kor - a_array[:, point[0], point[1]]) ** 2))
    #         rmse_Kor[1] += math.sqrt(sum((b_Kor - b_array[:, point[0], point[1]]) ** 2))
    #
    #     points_amount = points_i_amount * points_j_amount
    #     print(f'RMSE Bel: {rmse_Bel[0]/points_amount:.4f}, {rmse_Bel[1]/points_amount:.4f}, quantiles = {quantiles_amount}')
    #     print(f'RMSE Kor: {rmse_Kor[0] / points_amount:.4f}, {rmse_Kor[1] / points_amount:.4f}, quantiles = {quantiles_amount}')
    #     rmse_a_bel[q] = rmse_Bel[0] / points_amount
    #     rmse_b_bel[q] = rmse_Bel[1] / points_amount
    #     rmse_a_kor[q] = rmse_Kor[0] / points_amount
    #     rmse_b_kor[q] = rmse_Kor[1] / points_amount
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_A_bel.npy', rmse_a_bel)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_B_bel.npy', rmse_b_bel)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_A_kor.npy', rmse_a_kor)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_B_kor.npy', rmse_b_kor)
    #
    # plot_quantiles_amount_compare(files_path_prefix, 'A', quantiles)
    # plot_quantiles_amount_compare(files_path_prefix, 'B', quantiles)