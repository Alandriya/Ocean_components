import gc
from struct import unpack
import datetime
import numpy as np
import tqdm
import xarray as xr

# files_path_prefix = 'D://Data/OceanFull/'
files_path_prefix = 'D://Nastya/Data/OceanFull/'
# data_postfix = '2022_sst+press'
# data_postfix = '1979-1989_sst+press'
# data_postfix = 'sensible_2024_2025part'
#
# maskfile = open(files_path_prefix + "DATA/mask", "rb")
# binary_values = maskfile.read(29141)
# maskfile.close()
# mask = unpack('?' * 29141, binary_values)
# mask = np.array(mask, dtype=int)
# # mask = mask.reshape((161, 181))
#
# # variable_name = 'sst'
# # variable_name = 'sp'
# variable_name = 'sshf'
# # load grib
# # ds = xr.open_dataset(files_path_prefix + f'GRIB/{data_postfix}.grib', engine='cfgrib',
# #                      backend_kwargs={'filter_by_keys': {'shortName': variable_name}})
# ds = xr.open_dataset(files_path_prefix + f'GRIB/{data_postfix}.grib')
# # print(ds.variables)
# # raise ValueError
# # print(ds.coords['valid_time'][:10])
# # raise ValueError
# # print(ds.coords['time'].shape)
# # raise ValueError
#
#
# tmp = ds.variables[variable_name]
# # gc.collect()
#
# if variable_name in ['sst', 'sp']:
#     tmp = tmp[:, ::2, ::2]
#     shape = tmp.shape
#     print(tmp.shape)
#     tmp_new = tmp.data.reshape((-1, shape[1] * shape[2])).transpose()
#     del tmp
#     tmp_new = np.array(tmp_new)
#     print(tmp_new.shape)
#     tmp_new[np.logical_not(mask)] = np.nan
#     np.save(files_path_prefix + f'DATA/{data_postfix}.npy', tmp_new)
#
#     # data_numpy = np.zeros((161, 181, shape[0]))
#     #
#     # for t in tqdm.tqdm(range(0, tmp.shape[0])):
#     #     data_numpy[:, :, t] = tmp[t, :, :].data
#     #
#     # del tmp
#     # data_numpy = data_numpy.reshape((-1, data_numpy.shape[2]))
#     # print(data_numpy.shape)
#     # data_numpy[np.logical_not(mask), :] = np.nan
# else:
#     tmp = tmp[:, :, ::2, ::2]  # part of map with size 161x181
#     shape = tmp.shape
#     print(tmp.shape)
#     data_numpy = np.zeros((161, 181, shape[0] * shape[1]))
#
#     for t in tqdm.tqdm(range(0, tmp.shape[0])):
#         for i in range(161):
#             data_numpy[i, :, t * shape[1]:(t+1)*shape[1]] = tmp[t, :, i, :].data.transpose()
#
#     del tmp
#     data_numpy = data_numpy.reshape((-1, data_numpy.shape[2]))
#     print(data_numpy.shape)
#     np.save(files_path_prefix + f'Fluxes/{variable_name}_{data_postfix}.npy', data_numpy)
#     data_numpy[np.logical_not(mask)] = np.nan
#
#     np.save(files_path_prefix + f'Fluxes/{variable_name}_{data_postfix}.npy', data_numpy)

# for year in [2022]:
#     data_postfix = f'sst+press_{year}'
#     print(year)
#     if not os.path.exists(files_path_prefix + f'GRIB/{data_postfix}.grib'):
#         print(f'{year} missing')
#         continue
#     ds = xr.open_dataset(files_path_prefix + f'GRIB/{data_postfix}.grib', cache=True)
#
#     variable_name = 'sst'
#     tmp = ds.variables[variable_name]
#     tmp = tmp[:, ::2, ::2]
#     shape = tmp.shape
#     tmp_new = tmp.data.reshape((-1, shape[1] * shape[2])).transpose()
#     del tmp
#     tmp_new = np.array(tmp_new)
#     tmp_new[np.logical_not(mask), :] = np.nan
#     np.save(files_path_prefix + f'SST/SST_{year}.npy', tmp_new)
#
#     variable_name = 'sp'
#     tmp = ds.variables[variable_name]
#     tmp = tmp[:, ::2, ::2]
#     shape = tmp.shape
#     tmp_new = tmp.data.reshape((-1, shape[1] * shape[2])).transpose()
#     del tmp
#     tmp_new = np.array(tmp_new)
#     tmp_new[np.logical_not(mask), :] = np.nan
#     np.save(files_path_prefix + f'Pressure/PRESS_{year}.npy', tmp_new)


days_delta1 = (datetime.datetime(2025, 1, 1, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
days_delta2 = (datetime.datetime(2025, 11, 1, 0, 0) - datetime.datetime(2025, 1, 1, 0, 0)).days
print(days_delta1)
print(days_delta2)

# sst = np.load(files_path_prefix + f'DATA/SST_2024_2025part.npy')
# print(sst.shape)
# sst_2024 = np.zeros((sst.shape[0], days_delta1 * 4))
# sst_2025 = np.zeros((sst.shape[0], days_delta2 * 4))
# sst_2024[:, :] = sst[:, :days_delta1*4]
# sst_2025[:, :] = sst[:, days_delta1*4:(days_delta1 + days_delta2)*4]
#
# np.save(files_path_prefix + 'SST/SST_2024.npy', sst_2024)
# np.save(files_path_prefix + 'SST/SST_2025.npy', sst_2025)

# press = np.load(files_path_prefix + f'DATA/PRESS_2024_2025part.npy')
# print(press.shape)
# press_2024 = np.zeros((press.shape[0], days_delta1 * 4))
# press_2025 = np.zeros((press.shape[0], days_delta2 * 4))
# press_2024[:, :] = press[:, :days_delta1*4]
# press_2025[:, :] = press[:, days_delta1*4:(days_delta1 + days_delta2)*4]
#
# np.save(files_path_prefix + 'Pressure/PRESS_2024.npy', press_2024)
# np.save(files_path_prefix + 'Pressure/PRESS_2025.npy', press_2025)

fluxtype = 'latent'
flux = np.load(files_path_prefix + f'DATA/{fluxtype}_2024_2025part.npy')
print(flux.shape)
flux_2024 = np.zeros((flux.shape[0], days_delta1 * 4))
flux_2025 = np.zeros((flux.shape[0], days_delta2 * 4))
flux_2024[:, :] = flux[:, :days_delta1*4]
flux_2025[:, :] = flux[:, days_delta1*4:(days_delta1 + days_delta2)*4]

np.save(files_path_prefix + f'Fluxes/{fluxtype}_2024.npy', flux_2024)
np.save(files_path_prefix + f'Fluxes/{fluxtype}_2025.npy', flux_2025)