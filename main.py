import numpy as np
import scipy.linalg

from Plotting.plot_Bel_coefficients import *
from SRS_count_coefficients import *
from Plotting.mean_year import *
from Plotting.video import *
from Plotting.plot_eigenvalues import plot_eigenvalues
from extreme_evolution import *
from ABCF_coeff_counting import *
from eigenvalues import *
from data_processing import load_ABCFE, load_prepare_fluxes

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
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # count eigenvalues
    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2023_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_2019-2023_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2023_grouped.npy')

    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_1979-1989_grouped.npy')
    # SST_array = np.load(files_path_prefix + f'SST/SST_1979-1989_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_1979-1989_grouped.npy')
    #
    # t = 0
    # cpu_amount = 4
    #
    # n_bins = 100
    # offset = 0
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_flux_{n_bins}.npy', quantiles_flux)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_sst_{n_bins}.npy', quantiles_sst)
    # np.save(files_path_prefix + f'Eigenvalues\quantiles_press_{n_bins}.npy', quantiles_press)

    # count_eigenvalues_triplets(files_path_prefix, 0, flux_array, SST_array, press_array, mask, offset, n_bins, 1)
    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Flux', 'SST'), ('Flux', 'Pressure')]:
    #     count_mean_year(files_path_prefix, 1979, 2022, names, mask.reshape((161, 181)))

    # pair_name = 'Flux-Flux'
    # pair_name = 'Flux-SST'
    # pair_name = 'Flux-Pressure'
    # pair_name = 'SST-SST'
    # create_video(files_path_prefix, f'videos/Eigenvalues/{pair_name}/', f'Lambdas_', f'{pair_name}_eigenvalues', start=14610)


