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
timesteps = 20

if __name__ == '__main__':
    # delta = (end - start + cpu_count // 2) // cpu_count
    # print(end - start)

    # maskfile = open(files_path_prefix + "mask", "rb")
    # binary_values = maskfile.read(29141)
    # maskfile.close()
    # mask = unpack('?' * 29141, binary_values)

    parallel_AB(4)

    # binary_to_array(files_path_prefix, flux_type, "l79-21")

    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)
    # ---------------------------------------------------------------------------------------
    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)

    # a_timelist, b_timelist, borders = count_A_B_coefficients(files_path_prefix, mask, timesteps)
    # save_AB(files_path_prefix, a_timelist, b_timelist, None)

    # a_timelist, b_timelist, c_timelist, borders = load_AB(files_path_prefix, timesteps, load_c=True)
    # c_timelist = count_correlations(a_timelist, b_timelist)
    # save_AB(files_path_prefix, a_timelist, b_timelist, c_timelist)

    # a_timelist, b_timelist, c_timelist, borders = load_AB(files_path_prefix, timesteps)
    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, c_timelist, borders, timesteps)

    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'a_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'b_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'C_', 'c_daily', 10)