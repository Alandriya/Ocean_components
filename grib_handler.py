# import cfgrib
import tqdm
import xarray as xr
import numpy as np
import datetime
from data_processing import load_prepare_fluxes
from struct import unpack

files_path_prefix = 'D://Data/OceanFull/'
data_postfix = '2021-2022'

# load grib
ds = xr.open_dataset(files_path_prefix + f'{data_postfix}.grib', engine='cfgrib')
print(ds)
print(ds.coords['valid_time'])
# print(ds.coords['valid_time'].shape)
# print((datetime.datetime(2022, 4, 2) - datetime.datetime(2021, 1, 1)).days * 4)
raise ValueError
# print(ds.coords['longitude'])
# print(ds.variables['sshf'].values)
# raise ValueError

maskfile = open(files_path_prefix + "mask", "rb")
binary_values = maskfile.read(29141)
maskfile.close()
mask = unpack('?' * 29141, binary_values)
mask = np.array(mask, dtype=int)

# # sensible
# shape = ds.variables['msshf'].shape
# tmp = ds.variables['msshf'].values
# tmp = tmp[:, :, ::2, ::2]
# sensible = np.zeros((161*181, (shape[0] - 1) * shape[1]))
# for t in tqdm.tqdm(range(shape[0] - 1)):
#     for t1 in range(2):
#         for i in range(161):
#             for j in range(181):
#                 sensible[i*181 + j, t*2 + t1] = tmp[t, t1, i, j]
#
# # sensible = np.array(tmp).reshape(161*181, -1)
# sensible[np.logical_not(mask), :] = np.nan
# np.save(files_path_prefix + f'SENSIBLE_{data_postfix}.npy', sensible)

# # latent
# shape = ds.variables['mslhf'].shape
# tmp = ds.variables['mslhf'].values
# tmp = tmp[:, :, ::2, ::2]
# latent = np.zeros((161*181, (shape[0] - 1) * shape[1]))
# for t in tqdm.tqdm(range(shape[0] - 1)):
#     for t1 in range(2):
#         for i in range(161):
#             for j in range(181):
#                 latent[i*181 + j, t*2 + t1] = tmp[t, t1, i, j]
#
# latent[np.logical_not(mask), :] = np.nan
# np.save(files_path_prefix + f'LATENT_{data_postfix}.npy', latent)

# # group by day
# sensible_array, latent_array = load_prepare_fluxes(f'SENSIBLE_{data_postfix}.npy',
#                                                    f'LATENT_{data_postfix}.npy',
#                                                    prepare=False)
#
# np.save(files_path_prefix + f'sensible_grouped_{data_postfix}.npy', sensible_array)
# np.save(files_path_prefix + f'latent_grouped_{data_postfix}.npy', latent_array)


days_delta2 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days * 4
new_sensible = np.zeros((161*181, days_delta2))


# check previous data
days_delta1 = (datetime.datetime(2021, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days * 4 + 8
sens_old = np.load(files_path_prefix + 'SENSIBLE_2019-2021.npy')
sens_new = np.load(files_path_prefix + f'SENSIBLE_{data_postfix}.npy')

new_sensible[:, :days_delta1] = sens_old[:, 0:days_delta1]
new_sensible[:, days_delta1:] = sens_new[:, 10:sens_new.shape[1]-2]
np.save(files_path_prefix + 'SENSIBLE(01.01.2019 - 02.04.2022).npy', new_sensible)

print(sens_new[39, 10:110])
print(sens_old[39, days_delta1:days_delta1+10] == sens_new[39, 10:20])
print(new_sensible[39, days_delta1-10: days_delta1+10])

# lat_old = np.load(files_path_prefix + 'LATENT_2019-2021.npy')
# lat_new = np.load(files_path_prefix + f'LATENT_{data_postfix}.npy')
# # print(lat_old[39, days_delta1:days_delta1+8+10] == lat_new[39, 0:10])
# print(lat_old[39, days_delta1 - 10:days_delta1 -10 + 100])
# print(lat_new[39, 10:110])
#
#
# new_latent = np.zeros((161*181, days_delta2))
#
# new_latent[:, :days_delta1] = lat_old[:, 0:days_delta1]
# new_latent[:, days_delta1:] = lat_new[:, 10:lat_new.shape[1]-2]
# np.save(files_path_prefix + 'LATENT(01.01.2019 - 02.04.2022).npy', new_latent)
# print(new_latent[39, days_delta1-10: days_delta1+10])


