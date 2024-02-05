import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

x_label_list = ['90W', '60W', '30W', '0']
y_label_list = ['EQ', '30N', '60N', '80N']
xticks = [0, 60, 120, 180]
yticks = [160, 100, 40, 0]


def plot_eigenvalues(files_path_prefix: str,
                     matrix_list: list,
                     t: int,
                     lambdas_list: list,
                     names: tuple = ('Sensible', 'Latent'),
                     lambda_amount: int = 3,
                     shape: tuple = (161, 181),
                     ):
    fig, axs = plt.subplots(1, lambda_amount, figsize=(15, 5))
    img = [None for _ in range(lambda_amount)]

    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues')
    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}')

    cmap_l = plt.get_cmap('Blues').copy()
    cmap_l.set_bad('lightgreen', 1.0)

    date = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t)
    fig.suptitle(f'Lambdas {names[0]}-{names[1]}\n {date.strftime("%Y-%m-%d")}', fontsize=20)
    for i in range(lambda_amount):
        axs[i].set_title(f'Lambda {i + 1} = {lambdas_list[i]:.2f}', fontsize=12)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.3)
        lambda_matrix = matrix_list[i].reshape(shape)
        img[i] = axs[i].imshow(lambda_matrix,
                               interpolation='none',
                               cmap=cmap_l,
                               )
        axs[i].set_xticks(xticks)
        axs[i].set_yticks(yticks)
        axs[i].set_xticklabels(x_label_list)
        axs[i].set_yticklabels(y_label_list)
        fig.colorbar(img[i], cax=cax, orientation='vertical')

    # fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}/Lambdas_{t}.png')
    return
