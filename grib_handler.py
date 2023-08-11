import cfgrib
import tqdm
import xarray as xr
import numpy as np
from struct import unpack
import datetime

# files_path_prefix = 'D://Data/OceanFull/'
files_path_prefix = 'E://Nastya/Data/GRIB/'
# data_postfix = '2022-part'
data_postfix = '2'

maskfile = open('D://Data/OceanFull/' + "mask", "rb")
binary_values = maskfile.read(29141)
maskfile.close()
mask = unpack('?' * 29141, binary_values)
mask = np.array(mask, dtype=int)

# load grib
# ds = xr.open_dataset(files_path_prefix + f'{data_postfix}.grib', engine='cfgrib', drop_variables='mslhf', chunks=-1,
#                      inline_array=True)
ds = xr.open_dataset(files_path_prefix + f'{data_postfix}.grib', engine='cfgrib')

print(ds)
print(ds.coords['valid_time'])
# print(ds.coords['time'].shape)
raise ValueError

# sensible
shape = ds.variables['msshf'].shape
print(shape)
raise ValueError
# tmp = ds.variables['msshf']
tmp = ds.variables['mslhf'] #TODO latent
tmp = tmp[:, :, ::2, ::2]  # part of map with size 161x181
print(tmp.shape)



# print(tmp[370, :, 39, 39].values)  # ok
# print(tmp[400, :, 39, 39])  # OSError
# Если не упал до сих пор, то уже успех, но не полный
# raise ValueError


# sensible = np.zeros((161*181, shape[0] * shape[1]))
latent = np.zeros((161*181, shape[0] * shape[1]))
# print(sensible.shape)
for t in tqdm.tqdm(range(0, tmp.shape[0])):
    for i in range(161):
        # падает на следующей строчке при индексе >= 386, если не упадет, то надо раскомментить нормальный цикл с 0
        # sensible[i * 181:(i + 1) * 181, t * shape[1]:(t+1)*shape[1]] = tmp[t, :, i, :].transpose().values
        latent[i * 181:(i + 1) * 181, t * shape[1]:(t + 1) * shape[1]] = tmp[t, :, i, :].transpose().values

# sensible[np.logical_not(mask), :] = np.nan
latent[np.logical_not(mask), :] = np.nan

days_delta1 = (datetime.datetime(2022, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
days_delta2 = (datetime.datetime(2022, 9, 30, 0, 0) - datetime.datetime(2022, 1, 1, 0, 0)).days
# sensible_2019 = np.load(files_path_prefix + f'SENSIBLE_2019-2022.npy')
latent_2019 = np.load(files_path_prefix + f'LATENT_2019-2022.npy')

# full_sensible = np.zeros((29141, days_delta1*4 + (days_delta2+1)*4))
# full_sensible[:, 0:days_delta1*4] = sensible_2019[:, 0:days_delta1*4]
# full_sensible[:, days_delta1*4:] = sensible
# np.save(files_path_prefix + f'SENSIBLE_2019-2022[Oct].npy', full_sensible)
# np.save(files_path_prefix + f'SENSIBLE_{data_postfix}.npy', sensible)

full_latent = np.zeros((29141, days_delta1*4 + (days_delta2+1)*4))
full_latent[:, 0:days_delta1*4] = latent_2019[:, 0:days_delta1*4]
full_latent[:, days_delta1*4:] = latent
np.save(files_path_prefix + f'LATENT_2019-2022[Oct].npy', full_latent)
np.save(files_path_prefix + f'LATENT_{data_postfix}.npy', latent)
raise ValueError

# # latent
# shape = ds.variables['mslhf'].shape
# tmp = ds.variables['mslhf']
# tmp = tmp[:, :, ::2, ::2]
# # part = 100
# latent = np.zeros((161*181, part * shape[1]))
# for t in tqdm.tqdm(range(part)):
#     for t1 in range(12):
#         for i in range(161):
#             latent[i * 181:(i+1)*181, t * 2 + t1] = tmp[t, t1, i, :]
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


# days_delta2 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days * 4
# new_sensible = np.zeros((161*181, days_delta2))


# # check previous data
# days_delta1 = (datetime.datetime(2021, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days * 4 + 8
# sens_old = np.load(files_path_prefix + 'SENSIBLE_2019-2021.npy')
# sens_new = np.load(files_path_prefix + f'SENSIBLE_{data_postfix}.npy')
#
# new_sensible[:, :days_delta1] = sens_old[:, 0:days_delta1]
# new_sensible[:, days_delta1:] = sens_new[:, 10:sens_new.shape[1]-2]
# np.save(files_path_prefix + 'SENSIBLE(01.01.2019 - 02.04.2022).npy', new_sensible)
#
# print(sens_new[39, 10:110])
# print(sens_old[39, days_delta1:days_delta1+10] == sens_new[39, 10:20])
# print(new_sensible[39, days_delta1-10: days_delta1+10])

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


