import numpy as np

from Plotting.plot_Bel_coefficients import *
from SRS_count_coefficients import *
from Plotting.mean_year import *
from Plotting.video import *
from extreme_evolution import *
from ABCF_coeff_counting import *
from eigenvalues import *

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
    # pair_name = 'sst-press'
    # coeff_name = 'A'
    # start = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # create_video(files_path_prefix, f'videos/3D/{pair_name}/{coeff_name}/', f'{coeff_name}_', f'{pair_name}_{coeff_name}_2019-2023', 20, start)

    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    n_bins = 25
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019_grouped_scaled100.npy')
    SST_array = np.load(files_path_prefix + f'SST/SST_2019_grouped_scaled100.npy')
    press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019_grouped_scaled100.npy')
    values_flux = np.unique(flux_array)
    values_flux = values_flux[~numpy.isnan(values_flux)]
    values_sst = np.unique(SST_array)
    values_sst = values_sst[~numpy.isnan(values_sst)]
    values_press = np.unique(press_array)
    values_press = values_press[~numpy.isnan(values_press)]
    count_eigenvalues_triplets(files_path_prefix, mask, flux_array, values_flux, SST_array, values_sst,
                               press_array, values_press, offset, n_bins)