import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Plotting.video import get_continuous_cmap
import datetime
import tqdm
import gc
import os


x_label_list = ['90W', '60W', '30W', '0']
y_label_list = ['EQ', '30N', '60N', '80N']
xticks = [0, 60, 120, 180]
yticks = [160, 100, 40, 0]


def plot_eigenvals(files_path_prefix: str,
                   lambda_timelist: list,
                   borders: list,
                   time_start: int,
                   time_end: int,
                   start_pic_num: int = 0,
                   names: tuple = ('Sensible', 'Latent'),
                   path_local: str = '',
                   start_date: datetime = datetime.datetime(1979, 1, 1, 0, 0)
                   ):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    img = [None for _ in range(4)]
    lambda_max = borders[0]
    lambda_min = borders[1]

    if not os.path.exists(files_path_prefix + f'videos/{path_local}L'):
        os.mkdir(files_path_prefix + f'videos/{path_local}L')

    cmap_l1 = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, - lambda_min[0] / (lambda_max[0] - lambda_min[0]), 1])
    cmap_l1.set_bad('darkgreen', 1.0)

    cmap_l2 = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, - lambda_min[1] / (lambda_max[1] - lambda_min[1]), 1])
    cmap_l2.set_bad('darkgreen', 1.0)

    axs[0].set_title(f'Lambda - {names[0]}', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax1 = divider.append_axes('right', size='5%', pad=0.3)

    axs[1].set_title(f'Lambda - {names[1]}', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax2 = divider.append_axes('right', size='5%', pad=0.3)

    pic_num = start_pic_num
    step = 1
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = start_date + datetime.timedelta(days=start_pic_num + (t - time_start))
        fig.suptitle(f'Lambdas \n {date.strftime("%Y-%m-%d")}', fontsize=30)
        lambda_matrix = lambda_timelist[t]
        fig.suptitle(f'Lambda\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img[0] is None:
            img[0] = axs[0].imshow(lambda_matrix[0],
                                        interpolation='none',
                                        cmap=cmap_l1,
                                        vmin=lambda_min[0],
                                        vmax=lambda_max[0])
            axs[0].set_xticks(xticks)
            axs[0].set_yticks(yticks)
            axs[0].set_xticklabels(x_label_list)
            axs[0].set_yticklabels(y_label_list)
        else:
            img[0].set_data(lambda_matrix[0])

        fig.colorbar(img[0], cax=cax1, orientation='vertical')

        if img[1] is None:
            img[1] = axs[1].imshow(lambda_matrix[1],
                                        interpolation='none',
                                        cmap=cmap_l2,
                                        vmin=lambda_min[1],
                                        vmax=lambda_max[1])
            axs[1].set_xticks(xticks)
            axs[1].set_yticks(yticks)
            axs[1].set_xticklabels(x_label_list)
            axs[1].set_yticklabels(y_label_list)
        else:
            img[1].set_data(lambda_matrix[1])
        fig.colorbar(img[1], cax=cax2, orientation='vertical')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/{path_local}L/L_{pic_num:05d}.png')
        pic_num += 1

        del lambda_matrix
        gc.collect()
    return
