import gc
from struct import unpack

import numpy as np
import tqdm
import xarray as xr

# files_path_prefix = 'D://Data/OceanFull/'
files_path_prefix = 'D://Nastya/Data/OceanFull/'
# data_postfix = '2022_sst+press'
# data_postfix = '1979-1989_sst+press'
data_postfix = 'LATENT_2024_hourly'

maskfile = open(files_path_prefix + "mask", "rb")
binary_values = maskfile.read(29141)
maskfile.close()
mask = unpack('?' * 29141, binary_values)
mask = np.array(mask, dtype=int)
mask = mask.reshape((161, 181))

# variable_name = 'sst'
# variable_name = 'sp'
variable_name = 'avg_slhtf'
# load grib
# ds = xr.open_dataset(files_path_prefix + f'GRIB/{data_postfix}.grib', engine='cfgrib',
#                      backend_kwargs={'filter_by_keys': {'shortName': variable_name}})
ds = xr.open_dataset(files_path_prefix + f'GRIB/{data_postfix}.grib')
print(ds.variables)
# raise ValueError
# print(ds.coords['valid_time'])
# raise ValueError
# print(ds.coords['time'].shape)
# raise ValueError

# variable_name = 'sst'
# variable_name = 'mslhf'
tmp = ds.variables[variable_name]
gc.collect()

if variable_name in ['sst', 'sp']:
    tmp = tmp[:, ::2, ::2]
    shape = tmp.shape
    # print(tmp.shape)
    # tmp_new = tmp.data.reshape((-1, shape[1] * shape[2])).transpose()
    # del tmp
    # tmp_new = np.array(tmp_new)
    # tmp[:, np.where(np.logical_not(mask))[0], np.where(np.logical_not(mask))[1]] = np.nan
    np.save(files_path_prefix + f'DATA/{data_postfix}.npy', tmp)

    # data_numpy = np.zeros((161, 181, shape[0]))
    #
    # for t in tqdm.tqdm(range(0, tmp.shape[0])):
    #     data_numpy[:, :, t] = tmp[t, :, :].data
    #
    # del tmp
    # data_numpy = data_numpy.reshape((-1, data_numpy.shape[2]))
    # print(data_numpy.shape)
    # data_numpy[np.logical_not(mask), :] = np.nan
else:
    tmp = tmp[:, :, ::2, ::2]  # part of map with size 161x181
    shape = tmp.shape
    print(tmp.shape)
    data_numpy = np.zeros((161, 181, shape[0] * shape[1]))

    for t in tqdm.tqdm(range(0, tmp.shape[0])):
        for i in range(161):
            data_numpy[i, :, t * shape[1]:(t+1)*shape[1]] = tmp[t, :, i, :].data.transpose()

    # del tmp
    # data_numpy = data_numpy.reshape((-1, data_numpy.shape[2]))
    # print(data_numpy.shape)
    # data_numpy[np.logical_not(mask), :] = np.nan

    np.save(files_path_prefix + f'Fluxes/{variable_name}_{data_postfix}.npy', data_numpy)

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