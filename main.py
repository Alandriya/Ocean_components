import time
from EM_staff import *
from video import *
from data_processing import *
from ABC_coeff_counting import *
from func_estimation import *

# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
# flux_type = 'latent'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # parallel_AB(4)

    # ---------------------------------------------------------------------------------------
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    sensible_array = np.load(files_path_prefix + f'5years_sensible.npy')
    latent_array = np.load(files_path_prefix + f'5years_latent.npy')

    sensible_array = sensible_array.astype(float)
    latent_array = latent_array.astype(float)
    sensible_array[np.logical_not(mask), :] = np.nan
    latent_array[np.logical_not(mask)] = np.nan

    # mean by day = every 4 observations
    pack_len = 4
    sensible_array = block_reduce(sensible_array,
                                  block_size=(1, pack_len),
                                  func=np.mean, )
    latent_array = block_reduce(latent_array,
                                block_size=(1, pack_len),
                                func=np.mean, )

    sensible_array = scale_to_bins(sensible_array)
    latent_array = scale_to_bins(latent_array)

    # binary_to_array(files_path_prefix, flux_type, "l79-21")
    # ---------------------------------------------------------------------------------------
    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)

    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)
    # ---------------------------------------------------------------------------------------
    time_start = 0
    time_end = 5
    plot_step = 1

    # count_abf_coefficients(files_path_prefix, mask, sensible_array, latent_array, time_start, time_end)
    a_timelist, b_timelist, _, f_timelist, borders = load_ABCF(files_path_prefix, time_start + 1, time_end)
    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, time_start, time_end-1, plot_step)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, time_start, time_end-1, plot_step)

    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, 1, 3)
    # a_timelist, b_timelist, c_timelist, f_timelist, borders = load_ABCF(files_path_prefix, time_start + 1, time_end, load_c=True)
    # print('lol')
    # plot_c_coeff(files_path_prefix, c_timelist, time_start, len(c_timelist), plot_step)

    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'a_', 'a_weekly', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'b_', 'b_weekly', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'C_', 'c_weekly', 10)

    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)
    # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)

    plot_a_by_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, 0, len(a_timelist))