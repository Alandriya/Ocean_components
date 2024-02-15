import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Plotting.video import get_continuous_cmap

x_label_list = ['90W', '60W', '30W', '0']
y_label_list = ['EQ', '30N', '60N', '80N']
xticks = [0, 60, 120, 180]
yticks = [160, 100, 40, 0]


def plot_eigenvalues(files_path_prefix: str,
                     matrix,
                     t: int,
                     names: tuple = ('Sensible', 'Latent'),
                     shape: tuple = (161, 181),
                     ):
    fig, axs = plt.subplots(figsize=(5, 5))

    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues')
    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}')

    if names == ('Flux', 'SST'):
        cmap_l = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                     [0, - np.nanmin(matrix) / (np.nanmax(matrix) - np.nanmin(matrix)), 1])
    else:
        cmap_l = plt.get_cmap('Blues').copy()
    cmap_l.set_bad('lightgreen', 1.0)

    date = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t)
    fig.suptitle(f'Lambdas {names[0]}-{names[1]}\n {date.strftime("%Y-%m-%d")}', fontsize=20)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.3)
    lambda_matrix = matrix.reshape(shape)
    img = axs.imshow(lambda_matrix,
                     interpolation='none',
                     cmap=cmap_l,
                     vmin=np.nanmin(matrix),
                     vmax=np.nanmax(matrix)
                     )
    axs.set_xticks(xticks)
    axs.set_yticks(yticks)
    axs.set_xticklabels(x_label_list)
    axs.set_yticklabels(y_label_list)
    fig.colorbar(img, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}/Lambdas_{t}.png')
    return
