import numpy as np
from Plotting.plot_Bel_coefficients import *
from SRS_count_coefficients import *
from Plotting.mean_year import *
from Plotting.video import *
from Plotting.plot_eigenvalues import plot_eigenvalues
from extreme_evolution import *
from ABCF_coeff_counting import *
from eigenvalues import *
from data_processing import load_ABCFE

# files_path_prefix = 'home/aosipova/EM_ocean'
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
    # count eigenvalues
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2023_grouped.npy')
    SST_array = np.load(files_path_prefix + f'SST/SST_2019-2023_grouped.npy')
    press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2023_grouped.npy')
    t = 0
    cpu_amount = 4

    # flux_array = flux_array[:, t:t + 2]
    # SST_array = SST_array[:, t:t + 2]
    # press_array = press_array[:, t:t + 2]
    n_bins = 50

    # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)

    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    count_eigenvalues_triplets(files_path_prefix, flux_array, SST_array, press_array, mask, 0, offset, n_bins, cpu_amount)
    # count_eigenvalues_parralel(files_path_prefix, cpu_amount, flux_array, quantiles_flux, SST_array, quantiles_sst,
    #                        0, offset, ('Flux', 'SST'), n_bins)
    # count_eigenvalues_triplets(files_path_prefix, flux_array, SST_array, press_array, 0, offset, n_bins, 4)
    # t = 0
    # for names in [('Flux', 'SST'), ('Flux', 'Pressure'), ('SST', 'Pressure'), ('Flux', 'Flux'), ('SST', 'SST'),
    #               ('Pressure', 'Pressure')]:
    #     plot_eigenvalues(files_path_prefix, 3, mask, t + offset, names)
