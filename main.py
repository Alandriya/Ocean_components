import time
from EM_staff import *
from video import *
from data_processing import *
from ABC_coeff_counting import *

# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
# flux_type = 'latent'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # parallel_AB(4)

    # ---------------------------------------------------------------------------------------
    # maskfile = open(files_path_prefix + "mask", "rb")
    # binary_values = maskfile.read(29141)
    # maskfile.close()
    # mask = unpack('?' * 29141, binary_values)
    #
    # sensible_array = np.load(files_path_prefix + f'5years_sensible.npy')
    # latent_array = np.load(files_path_prefix + f'5years_latent.npy')
    #
    # sensible_array = sensible_array.astype(float)
    # latent_array = latent_array.astype(float)
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array[np.logical_not(mask)] = np.nan
    #
    # # mean by day = every 4 observations
    # pack_len = 4
    # sensible_array = block_reduce(sensible_array,
    #                               block_size=(1, pack_len),
    #                               func=np.mean, )
    # latent_array = block_reduce(latent_array,
    #                             block_size=(1, pack_len),
    #                             func=np.mean, )
    #
    # sensible_array = scale_to_bins(sensible_array)
    # latent_array = scale_to_bins(latent_array)
    #
    # count_A_B_coefficients(files_path_prefix, mask, sensible_array, latent_array, 0, 2)
    # ---------------------------------------------------------------------------------------

    # binary_to_array(files_path_prefix, flux_type, "l79-21")
    # ---------------------------------------------------------------------------------------
    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)

    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)
    # ---------------------------------------------------------------------------------------

    # a_timelist, b_timelist, borders = count_A_B_coefficients(files_path_prefix, mask, timesteps)
    # save_AB(files_path_prefix, a_timelist, b_timelist, None)

    a_timelist, b_timelist, c_timelist, borders = load_ABC(files_path_prefix, 1, 1829)
    # c_timelist = count_correlations(a_timelist, b_timelist)
    # save_AB(files_path_prefix, a_timelist, b_timelist, c_timelist)

    # a_timelist, b_timelist, c_timelist, borders = load_AB(files_path_prefix, timesteps)
    plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, c_timelist, borders, 1829)

    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'a_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'b_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'C_', 'c_daily', 10)