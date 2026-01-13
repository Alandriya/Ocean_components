import datetime
from struct import unpack

import numpy as np

from Data_processing.data_processing import *
from Data_processing.func_estimation import *
# from Plotting.plot_eigenvalues import plot_eigenvalues, plot_mean_year
# from Plotting.plot_extreme import *
# from extreme_evolution import *
# from ABCF_coeff_counting import *
from Eigenvalues.eigenvalues import *
from Plotting.plot_func_estimations import plot_ab_functional
# from Plotting.plot_Bel_coefficients import *
# from SRS_count_coefficients import *
# from Plotting.mean_year import *
from Plotting.video import *
from Coefficients.Kor_Bel_compare import *
from Forecasting.utils import fix_random
# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'D:/Nastya/Data/OceanFull/'

width = 181
height = 161
fix_random(2025)

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "DATA/mask", "rb")
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
    days_delta6 = (datetime.datetime(2025, 11, 1, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # estimate the functional a(X) and b(X) from data
    start_year = 1979
    end_year = 1989
    start = 0
    end = 1000
    start_index = 0
    data_name = 'latent'
    # sensible = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')[:, start: end+1]
    # # print(sensible.shape) # (29141, 3653)
    # sensible = sensible.transpose()
    # sensible = sensible.reshape((-1, height, width))
    # print(sensible.shape)
    latent = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')[:, start: end+1]
    latent = latent.transpose()
    latent = latent.reshape((-1, height, width))
    data_array = latent
    count_1d_Korolev(files_path_prefix,
                     data_array,
                     time_start=start,
                     time_end=end + 1,
                     path=f'Components/{data_name}/',
                     quantiles_amount=250,
                     n_components=2,
                     start_index=start_index)

    a_array = np.zeros_like(data_array)
    b_array = np.zeros_like(data_array)
    for t in tqdm.tqdm(range(data_array.shape[0]-1)):
        a_array[t] = np.load(files_path_prefix + f'Components/{data_name}/Kor/daily/A_{t+start_index+1}.npy')
        b_array[t] = np.load(files_path_prefix + f'Components/{data_name}/Kor/daily/B_{t+start_index+1}.npy')
    np.save(files_path_prefix + f'Components/{data_name}/a_{start_index}-{start_index + a_array.shape[0]}.npy', a_array)
    np.save(files_path_prefix + f'Components/{data_name}/b_{start_index}-{start_index + a_array.shape[0]}.npy', b_array)

    a_array = np.load(files_path_prefix + f'Components/{data_name}/a_{start_index}-{start_index + data_array.shape[0]}.npy')
    b_array = np.load(files_path_prefix + f'Components/{data_name}/b_{start_index}-{start_index + data_array.shape[0]}.npy')
    quantiles, a_grouped, b_grouped = estimate_A_B(files_path_prefix, data_array, a_array, b_array)
    np.save(files_path_prefix + f'Components/{data_name}/quantiles.npy', quantiles)
    np.save(files_path_prefix + f'Components/{data_name}/a_grouped.npy', a_grouped)
    np.save(files_path_prefix + f'Components/{data_name}/b_grouped.npy', b_grouped)

    quantiles = np.load(files_path_prefix + f'Components/{data_name}/quantiles.npy')
    a_grouped = np.load(files_path_prefix + f'Components/{data_name}/a_grouped.npy')
    b_grouped = np.load(files_path_prefix + f'Components/{data_name}/b_grouped.npy')
    # print(quantiles)
    # print(a_grouped)
    # print(b_grouped)
    plot_ab_functional(files_path_prefix, quantiles, a_grouped, b_grouped, data_name)
    # ----------------------------------------------------------------------------------------------