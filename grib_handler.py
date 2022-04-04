# import cfgrib
import tqdm
import xarray as xr
import numpy as np
import datetime
from data_processing import load_prepare_fluxes
from struct import unpack

files_path_prefix = 'D://Data/OceanFull/'
data_postfix = '2021'

# load grib
ds = xr.open_dataset(files_path_prefix + f'{data_postfix}.grib', engine='cfgrib')
# print(ds)
# print(ds.coords['longitude'])

maskfile = open(files_path_prefix + "mask", "rb")
binary_values = maskfile.read(29141)
maskfile.close()
mask = unpack('?' * 29141, binary_values)
mask = np.array(mask, dtype=int)

shape = ds.variables['msshf'].shape

# sensible
tmp = ds.variables['msshf'].values
sensible = np.zeros((161*181, shape[0] * shape[1]))
for t in tqdm.tqdm(range(shape[0])):
    for t1 in range(2):
        for i in range(161):
            for j in range(181):
                sensible[i*181 + j, t*2 + t1] = tmp[t, t1, i * 2, j * 2]

sensible[np.logical_not(mask), :] = np.nan
np.save(files_path_prefix + f'SENSIBLE_{data_postfix}.npy', sensible)

# latent
tmp = ds.variables['mslhf'].values
latent = np.zeros((161*181, shape[0] * shape[1]))
for t in tqdm.tqdm(range(shape[0])):
    for t1 in range(2):
        for i in range(161):
            for j in range(181):
                latent[i*181 + j, t*2 + t1] = tmp[t, t1, i * 2, j * 2]

latent[np.logical_not(mask), :] = np.nan
np.save(files_path_prefix + f'LATENT_{data_postfix}.npy', latent)

# group by day
sensible_array, latent_array = load_prepare_fluxes(f'SENSIBLE_{data_postfix}.npy',
                                                   f'LATENT_{data_postfix}.npy',
                                                   prepare=False)

np.save(files_path_prefix + f'sensible_grouped_{data_postfix}.npy', sensible_array)
np.save(files_path_prefix + f'latent_grouped_{data_postfix}.npy', latent_array)


# # check previous data
# # sens_old = np.load(files_path_prefix + 'sensible_grouped_2019-2021.npy')
# # sens_new = np.load(files_path_prefix + 'sensible_grouped_2021.npy')
# sens_old = np.load(files_path_prefix + 'SENSIBLE_2019-2021.npy')
# sens_new = np.load(files_path_prefix + 'SENSIBLE_2021.npy')
# days_delta1 = (datetime.datetime(2021, 7, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days * 4
# print(days_delta1)
# print(sens_old.shape)
# print(sens_new.shape)
# print(sens_old[39:40, days_delta1:days_delta1+15])
# print(sens_new[39:40, :15])
