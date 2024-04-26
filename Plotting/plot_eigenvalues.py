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
                     n_lambdas: int,
                     mask: np.ndarray,
                     t: int,
                     array1,
                     array2,
                     array1_quantiles,
                     array2_quantiles,
                     names: tuple = ('Sensible', 'Latent'),
                     shape: tuple = (161, 181),

                     ):
    eigenvalues = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t}.npy')
    eigenvalues = np.real(eigenvalues)
    eigenvalues = [0 if np.isnan(e) else e for e in eigenvalues]
    positions = [x for x in range(len(eigenvalues))]
    positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
    eigenvectors = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvectors_{t}.npy')

    width, height = shape
    matrix_list = [np.zeros(height * width) for _ in range(n_lambdas)]
    lambda_list = []
    max_list=[]
    min_list=[]

    n_bins=len(eigenvalues)

    for l in range(n_lambdas):
        # print(eigenvalues[positions[l]])
        # eigenvector = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvectors_{t}/{positions[l]}.npy')
        for j1 in range(0, n_bins):
            points_y1 = np.where((array2_quantiles[j1] <= array2[:, 0]) & (array2[:, 0] < array2_quantiles[j1 + 1]))[0]
            matrix_list[l][points_y1] = np.real(eigenvectors[j1, positions[l]])
        matrix_list[l][np.logical_not(mask)] = np.nan
        max_list.append(np.nanmax(matrix_list[l]))
        min_list.append(np.nanmin(matrix_list[l]))
        lambda_list.append(eigenvalues[positions[l]])

    fig, axs = plt.subplots(1, n_lambdas, figsize=(5*n_lambdas, 5))
    img = [None for _ in range(n_lambdas)]

    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues')
    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}')

    # cmap_l = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, ( - min(min_list)) / (max(max_list) - min(min_list), 1])
    # y_min = min(min_list)
    # y_max = max(max_list)
    # cmap_l = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - y_min) / (y_max - y_min), 1])

    cmap_l = plt.get_cmap('Blues').copy()
    cmap_l.set_bad('lightgreen', 1.0)

    date = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t)
    fig.suptitle(f'Lambdas {names[0]}-{names[1]}\n {date.strftime("%Y-%m-%d")}', fontsize=20)
    for i in range(n_lambdas):
        axs[i].set_title(f'Lambda {i + 1} = {lambda_list[i]:.2e}', fontsize=12)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.3)
        lambda_matrix = matrix_list[i].reshape(shape)
        img[i] = axs[i].imshow(lambda_matrix,
                               interpolation='none',
                               cmap=cmap_l,
                               vmin=0,
                               vmax=0.5,
                               )
        axs[i].set_xticks(xticks)
        axs[i].set_yticks(yticks)
        axs[i].set_xticklabels(x_label_list)
        axs[i].set_yticklabels(y_label_list)
        fig.colorbar(img[i], cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}/Lambdas_{t}.png')
    plt.close(fig)
    return
