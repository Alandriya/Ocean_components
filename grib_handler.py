# import cfgrib
import xarray as xr
import numpy as np
import datetime
from data_processing import load_prepare_fluxes

files_path_prefix = 'D://Data/OceanFull/'
data_postfix = '2021'

# load grib
ds = xr.open_dataset(files_path_prefix + f'{data_postfix}.grib', engine='cfgrib')
# print(ds)
# print(ds.coords['longitude'])

# latent
shape = ds.variables['mslhf'].shape
latent = np.array(ds.variables['mslhf']).reshape((shape[0]*shape[1], shape[2], shape[3]))
latent = latent[:, ::2, ::2]
print(latent.shape)  # [timesteps, 161, 181]
latent = latent.reshape((161 * 181, latent.shape[0]))
np.save(files_path_prefix + f'LATENT_{data_postfix}.npy', latent)

# sensible
shape = ds.variables['msshf'].shape
sensible = np.array(ds.variables['msshf']).reshape((shape[0]*shape[1], shape[2], shape[3]))
sensible = sensible[:, ::2, ::2]
print(sensible.shape)  # [timesteps, 161, 181]
sensible = np.transpose(sensible)
sensible = sensible.reshape((161*181, sensible.shape[2]))
print(sensible.shape)  # [161, 181, timesteps]
# sensible = sensible.reshape((161 * 181, sensible.shape[2]))
np.save(files_path_prefix + f'SENSIBLE_{data_postfix}.npy', sensible)

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
# sens_new = np.load(files_path_prefix + 'SENSIBLE_2021(jul-dec).npy')
# days_delta1 = (datetime.datetime(2021, 7, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days * 4
# print(days_delta1)
# print(sens_old.shape)
# print(sens_new.shape)
# print(sens_old[39:41, days_delta1:days_delta1+15])
# print(sens_new[39:41, :15])
