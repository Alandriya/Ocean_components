import datetime
import os.path
import time

import numpy as np
import pandas as pd
import scipy.stats
import tqdm

from video import *
from plot_fluxes import *
from plot_Bel_coefficients import *
from data_processing import *
from ABCF_coeff_counting import *
from Kor_Bel_compare import *
from func_estimation import *
from data_processing import load_prepare_fluxes
from func_estimation import estimate_a_flux_by_months
from extreme_evolution import *
import cycler
from EM_hybrid import *
from fluxes_distribution import *
from SRS_count_coefficients import *
from copy import deepcopy
import shutil
import pyswarms
from mean_year import *


# Parameters
files_path_prefix = 'D://Data/OceanFull/'

# timesteps = 7320
timesteps = 1829
width = 181
height = 161

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    borders = [[0, 1000]]
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2022, 9, 30, 0, 0) - datetime.datetime(2022, 4, 2, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # Plot differences of mean years
    coeff_type = 'B'
    flux_type = 'sensible'
    start_year = 2009
    end_year = 2019
    postfix = 'absolute_difference'
    mean_year_Bel = np.load(files_path_prefix + f'Mean_year/{coeff_type}_2009-2019_{flux_type}_Bel.npy')
    mean_year_Kor = np.load(files_path_prefix + f'Mean_year/{coeff_type}_2009-2019_{flux_type}_Kor.npy')
    mean_year = np.abs(mean_year_Kor - mean_year_Bel)
    coeff_min = np.nanmin(mean_year)
    coeff_max = np.nanmax(mean_year)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # fig.suptitle(f'{coeff_type} absolute differences {flux_type} mean year', fontsize=30)
    axs[0][0].title.set_text('February, 15')
    axs[0][1].title.set_text('April, 15')
    axs[0][2].title.set_text('June, 15')
    axs[1][0].title.set_text('August, 15')
    axs[1][1].title.set_text('October, 15')
    axs[1][2].title.set_text('December, 15')
    img = [None for _ in range(6)]
    cax = [None for _ in range(6)]
    days = [(datetime.datetime(start_year, i * 2, 15) - datetime.datetime(start_year, 1, 2)).days for i in range(1, 7)]

    # cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    cmap = plt.get_cmap('Reds')
    cmap.set_bad('darkgreen', 1.0)
    for i in range(6):
        divider = make_axes_locatable(axs[i // 3][i % 3])
        cax[i] = divider.append_axes('right', size='5%', pad=0.3)
        img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                           interpolation='none',
                                           cmap=cmap,
                                           vmin=coeff_min,
                                           vmax=coeff_max)

        x_label_list = ['90W', '60W', '30W', '0']
        y_label_list = ['EQ', '30N', '60N', '80N']
        xticks = [0, 60, 120, 180]
        yticks = [160, 100, 40, 0]

        axs[i // 3][i % 3].set_xticks(xticks)
        axs[i // 3][i % 3].set_yticks(yticks)
        axs[i // 3][i % 3].set_xticklabels(x_label_list)
        axs[i // 3][i % 3].set_yticklabels(y_label_list)
        fig.colorbar(img[i], cax=cax[i], orientation='vertical')

    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Mean_year/{coeff_type}_{flux_type}_{start_year}-{end_year}_{postfix}.png')
