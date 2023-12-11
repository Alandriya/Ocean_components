from Plotting.plot_Bel_coefficients import *
from SRS_count_coefficients import *
from Plotting.mean_year import *
from Plotting.video import *

# Parameters
# files_path_prefix = 'D://Data/OceanFull/'
from data_processing import load_ABCF

# files_path_prefix = 'E://Nastya/Data/OceanFull/'
files_path_prefix = 'D://Data/OceanFull/'

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


