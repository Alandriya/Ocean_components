import copy
import datetime
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
# files_path_prefix = 'D://Data/OceanFull/'
import seaborn as sns
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
    creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def plot_predictions(files_path_prefix: str,
                     Y_test: np.ndarray,
                     Y_predict: np.ndarray,
                     model_name: str,
                     features_amount: int,
                     start_day: datetime.datetime,
                     mask: np.ndarray,
                     cfg: None,
                     postfix_simple=True):
    # print('Plotting')
    # (5, 3, 81, 91)
    # sns.set_style("whitegrid")
    if not os.path.exists(files_path_prefix + f'videos/Forecast/{model_name}'):
        os.mkdir(files_path_prefix + f'videos/Forecast/{model_name}')

    days_prediction = Y_predict.shape[0]

    if not os.path.exists(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days'):
        os.mkdir(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days')

    Y_test[:, :, np.logical_not(mask)] = np.nan
    Y_predict[:, :, np.logical_not(mask)] = np.nan

    # axs[0].set_title('Real values', fontsize=20)
    # axs[1].set_title('Predicted values', fontsize=20)
    # axs[2].set_title('Absolute difference', fontsize=20)

    # flux_min, sst_min, press_min = cfg.min_vals[:3]
    # flux_max, sst_max, press_max = cfg.max_vals[:3]

    flux_min, sst_min, press_min = 0, 0, 0
    flux_max, sst_max, press_max = 1, 1, 1
    flux_min = min(np.nanmin(Y_test[:, 0]), np.nanmin(Y_predict[:, 0]))
    flux_max = max(np.nanmax(Y_test[:, 0]), np.nanmax(Y_predict[:, 0]))
    # flux_min = min(-1, flux_min)
    # flux_max = max(flux_max, 1)

    # sst_min = min(np.nanmin(Y_test[:, 1]), np.nanmin(Y_predict[:, 1]))
    # sst_max = max(np.nanmax(Y_test[:, 1]), np.nanmax(Y_predict[:, 1]))
    #
    # press_min = min(np.nanmin(Y_test[:, 2]), np.nanmin(Y_predict[:, 2]))
    # press_max = max(np.nanmax(Y_test[:, 2]), np.nanmax(Y_predict[:, 2]))
    # print(flux_min)
    # print(flux_max)
    # print(sst_min)
    # print(sst_max)
    # print(press_min)
    # print(press_max)

    cmap_flux = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                    [0, (1.0 - flux_min) / (flux_max - flux_min), 1])

    cmap_flux2 = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                    [0, (1.0 - np.nanmin(Y_predict[:, 0])) / (np.nanmax(Y_predict[:, 0]) - np.nanmin(Y_predict[:, 0])), 1])
    # cmap_flux = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    # cmap_flux = plt.get_cmap('Blues').copy()
    # cmap_flux = plt.get_cmap('plasma').copy()
    cmap_flux.set_bad('lightgreen', 1.0)
    cmap_flux2.set_bad('lightgreen', 1.0)
    # cmap_sst = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    # cmap_sst = plt.get_cmap('Oranges').copy()
    # cmap_sst = plt.get_cmap('Blues').copy()
    cmap_sst = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                    [0, (1.0 - sst_min) / (sst_max - sst_min), 1])
    # cmap_sst = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    cmap_sst.set_bad('lightgreen', 1.0)
    # cmap_press = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    # cmap_press = plt.get_cmap('Purples').copy()
    # cmap_press = plt.get_cmap('Blues').copy()
    cmap_press = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                    [0, (1.0 - press_min) / (press_max - press_min), 1])
    # cmap_press = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    cmap_press.set_bad('lightgreen', 1.0)

    cmap_diff = plt.get_cmap('Reds').copy()
    cmap_diff.set_bad('lightgreen', 1.0)

    x_label_list = ['90W', '60W', '30W', '0']
    y_label_list = ['EQ', '30N', '60N', '80N']
    xticks = [0, 30, 60, 90]
    yticks = [80, 50, 20, 0]

    # print(Y_test[0, 0])
    # print(Y_predict[0, 0])

    for k in range(cfg.channels):
        fig, axs = plt.subplots(3, days_prediction, figsize=(5 * days_prediction, 15))
        img = [[None for _ in range(days_prediction)] for _ in range(3)]
        cax = [[None for _ in range(days_prediction)] for _ in range(3)]
        if k == 0:
            cmap_test = copy.deepcopy(cmap_flux)
            cmap_pred = copy.deepcopy(cmap_flux2)
            # test_min = pred_min = flux_min
            # test_max = pred_max = flux_max
            test_min = np.nanmin(Y_test[:, 0])
            test_max = np.nanmax(Y_test[:, 0])
            pred_min = np.nanmin(Y_predict[:, 0])
            pred_max = np.nanmax(Y_predict[:, 0])
        elif k == 1:
            cmap_test = cmap_pred = copy.deepcopy(cmap_sst)
            test_min = pred_min = sst_min
            test_max = pred_max = sst_max
        else:
            cmap_test = cmap_pred = copy.deepcopy(cmap_press)
            test_min = pred_min = press_min
            test_max = pred_max = press_max

        for t in range(days_prediction):
            ypredict = Y_predict[t, k, :, :]
            ytest = Y_test[t, k, :, :]
            difference = np.array(np.abs(ypredict - ytest))
            day_str = (start_day + datetime.timedelta(days=t)).strftime('%d.%m.%Y')
            axs[0][t].set_title(f'{day_str}, real values', fontsize=16)
            axs[1][t].set_title(f'{day_str}, predictions', fontsize=16)
            axs[2][t].set_title(f'{day_str}, absolute difference', fontsize=16)
            for i in range(3):
                divider = make_axes_locatable(axs[i][t])
                cax[i][t] = divider.append_axes('right', size='5%', pad=0.3)
                axs[i][t].set_xticks(xticks)
                axs[i][t].set_yticks(yticks)
                axs[i][t].set_xticklabels(x_label_list)
                axs[i][t].set_yticklabels(y_label_list)

            img[0][t] = axs[0][t].imshow(ytest,
                                         interpolation='none',
                                         cmap=cmap_test,
                                         vmin=test_min,
                                         vmax=test_max)

            img[1][t] = axs[1][t].imshow(ypredict,
                                         interpolation='none',
                                         cmap=cmap_pred,
                                         vmin=pred_min,
                                         vmax=pred_max)

            img[2][t] = axs[2][t].imshow(difference,
                                         interpolation='none',
                                         cmap=cmap_diff,
                                         vmin=0,
                                         vmax=np.nanmax(np.abs(Y_predict[:, k] - Y_test[:Y_predict.shape[0], k]))/2)

            for i in range(3):
                fig.colorbar(img[i][t], cax=cax[i][t], orientation='vertical')

        if k == 0:
            fig.suptitle(f'{model_name}, Flux', fontsize=30)
        elif k == 1:
            fig.suptitle(f'{model_name}, SST', fontsize=30)
        else:
            fig.suptitle(f'{model_name}, Pressure', fontsize=30)
        plt.tight_layout()
        if postfix_simple:
            if k == 0:
                fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days/{model_name}_{features_amount}_Flux_{day_str}_simple.png')
            elif k == 1:
                fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days/{model_name}_{features_amount}_SST_{day_str}_simple.png')
            else:
                fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days/{model_name}_{features_amount}_press_{day_str}_simple.png')
        else:
            if k == 0:
                fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days/{model_name}_{features_amount}_Flux_{day_str}_informed.png')
            elif k == 1:
                fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days/{model_name}_{features_amount}_SST_{day_str}_informed.png')
            else:
                fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{days_prediction}_days/{model_name}_{features_amount}_press_{day_str}_informed.png')
        plt.close(fig)
    return


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_1d_predictions(files_path_prefix: str,
                        y_test: np.ndarray,
                        y_predict: np.ndarray,
                        model_name: str,
                        start_day: datetime.datetime,
                        cluster_idx: int,
                        point_idx: int,
                        ):
    fig = plt.figure(figsize=(10, 5))
    days_prediction = len(y_test)
    days_str = [(start_day + datetime.timedelta(days=d)).strftime('%d.%m.%Y') for d in range(days_prediction)]
    plt.plot(days_str, y_test, c='r', label='Test values')
    plt.plot(days_str, y_predict, '-o', c='b', label='Prediction')
    fig.suptitle(f'Prediction of point {point_idx} in cluster {cluster_idx}')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists(files_path_prefix + f'videos/Forecast/1d'):
        os.mkdir(files_path_prefix + f'videos/Forecast/1d')
    if not os.path.exists(files_path_prefix + f'videos/Forecast/1d/{model_name}'):
        os.mkdir(files_path_prefix + f'videos/Forecast/1d/{model_name}')
    if not os.path.exists(files_path_prefix + f'videos/Forecast/1d/{model_name}/cluster_{cluster_idx}'):
        os.mkdir(files_path_prefix + f'videos/Forecast/1d/{model_name}/cluster_{cluster_idx}')
    plt.savefig(files_path_prefix + f'videos/Forecast/1d/{model_name}/cluster_{cluster_idx}/{point_idx}-cluster_{cluster_idx}.png')
    plt.close(fig)
    return


def plot_clusters(files_path_prefix: str,
        frequencies: np.ndarray,
        labels: np.ndarray,
        yhat: np.ndarray,
        filename: str):
    fig, axs = plt.subplots(3, 3, figsize=(30, 30))
    for label in labels:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == label)
        # create scatter of these samples
        axs[0][0].scatter(frequencies[row_ix, 0], frequencies[row_ix, 1])
        axs[0][1].scatter(frequencies[row_ix, 0], frequencies[row_ix, 2])
        axs[0][2].scatter(frequencies[row_ix, 0], frequencies[row_ix, 3])
        axs[1][0].scatter(frequencies[row_ix, 1], frequencies[row_ix, 0])
        axs[1][1].scatter(frequencies[row_ix, 1], frequencies[row_ix, 2])
        axs[1][2].scatter(frequencies[row_ix, 1], frequencies[row_ix, 3])
        axs[2][0].scatter(frequencies[row_ix, 2], frequencies[row_ix, 0])
        axs[2][1].scatter(frequencies[row_ix, 2], frequencies[row_ix, 1])
        axs[2][2].scatter(frequencies[row_ix, 2], frequencies[row_ix, 3])

    # plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Forecast/Clusters/{filename}.png')
    return


def plot_train_loss(files_path_prefix, loss_arr, start_year, end_year, model_name):
    if not os.path.exists(files_path_prefix + f'videos/Forecast/Loss'):
        os.mkdir(files_path_prefix + f'videos/Forecast/Loss')
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f'{model_name} loss, training on {start_year} - {end_year}')
    x = np.linspace(1, len(loss_arr), len(loss_arr)//2, dtype=int)
    plt.plot(loss_arr, '-o', label='Train loss')
    plt.xticks(x)
    plt.xlabel('Iteration')
    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Forecast/Loss/{model_name}_{start_year}-{end_year}.png')
    return


def plot_flux_sst_press(files_path_prefix: str,
                        flux: np.ndarray,
                        sst: np.ndarray,
                        press: np.ndarray,
                        start: int,
                        end: int,
                        start_date: datetime.datetime = datetime.datetime(1979, 1, 1, 0, 0),
                        start_pic_num: int = 1,
                        frequency: int = 1,
                        ):
    sns.set_style("whitegrid")

    # if not os.path.exists(files_path_prefix + f'videos/3D/flux-sst-press/hourly'):
    #     os.mkdir(files_path_prefix + f'videos/3D/flux-sst-press/hourly')

    if not os.path.exists(files_path_prefix + f'videos/3D/flux-sst-press/correlations'):
        os.mkdir(files_path_prefix + f'videos/3D/flux-sst-press/correlations')

    if not os.path.exists(files_path_prefix + f'videos/3D/flux-sst-press/correlations/{frequency}'):
        os.mkdir(files_path_prefix + f'videos/3D/flux-sst-press/correlations/{frequency}')

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    img_flux, img_sst, img_press = None, None, None

    flux_max = np.nanmax(flux)
    flux_min = np.nanmin(flux)
    sst_max = np.nanmax(sst)
    sst_min = np.nanmin(sst)
    press_max = np.nanmax(press)
    press_min = np.nanmin(press)

    # cmap_flux = plt.get_cmap('Blues').copy()
    # cmap_flux = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
    cmap_flux =cmap_sst = cmap_press = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, 0.5, 1])
    cmap_flux.set_bad('lightgreen', 1.0)
    # cmap_sst = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    # cmap_sst = plt.get_cmap('Reds').copy()
    cmap_sst.set_bad('lightgreen', 1.0)
    # cmap_press = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    # cmap_press = plt.get_cmap('Purples').copy()
    cmap_press.set_bad('lightgreen', 1.0)

    # axs[0].set_title(f'Sum flux', fontsize=20)
    axs[0].set_title(f'Flux-SST', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax_flux = divider.append_axes('right', size='5%', pad=0.3)

    # axs[1].set_title(f'SST', fontsize=20)
    axs[1].set_title(f'SST-Pressure', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax_sst = divider.append_axes('right', size='5%', pad=0.3)

    # axs[2].set_title(f'Pressure', fontsize=20)
    axs[2].set_title(f'Flux-Pressure', fontsize=20)
    divider = make_axes_locatable(axs[2])
    cax_press = divider.append_axes('right', size='5%', pad=0.3)

    x_label_list = ['90W', '60W', '30W', '0']
    y_label_list = ['EQ', '30N', '60N', '80N']
    # xticks = [0, 60, 120, 180]
    # yticks = [160, 100, 40, 0]
    xticks = [0, 30, 60, 90]
    yticks = [80, 50, 20, 0]

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(start, end)):
        # if frequency == 1:
        date = start_date + datetime.timedelta(days=(t - start))
        date2 = date + datetime.timedelta(days=7)
        # fig.suptitle(f'Correlations {date.strftime("%Y-%m-%d")} - {date2.strftime("%Y-%m-%d")}', fontsize=30)

        # elif frequency == 24:
        #     date = start_date + datetime.timedelta(hours=(t - start))
        #     fig.suptitle(f'{date.strftime("%Y-%m-%d %H:00")}', fontsize=30)
        # fig.suptitle(f'{date.strftime("%Y-%m-%d")}', fontsize=30)

        if img_flux is None:
            img_flux = axs[0].imshow(flux[t],
                                     interpolation='none',
                                     cmap=cmap_flux,
                                     vmin=flux_min,
                                     vmax=flux_max)
            axs[0].set_xticks(xticks)
            axs[0].set_yticks(yticks)
            axs[0].set_xticklabels(x_label_list)
            axs[0].set_yticklabels(y_label_list)

            img_sst = axs[1].imshow(sst[t],
                                     interpolation='none',
                                     cmap=cmap_sst,
                                     vmin=sst_min,
                                     vmax=sst_max)
            axs[1].set_xticks(xticks)
            axs[1].set_yticks(yticks)
            axs[1].set_xticklabels(x_label_list)
            axs[1].set_yticklabels(y_label_list)

            img_press = axs[2].imshow(press[t],
                                     interpolation='none',
                                     cmap=cmap_press,
                                     vmin=press_min,
                                     vmax=press_max)
            axs[2].set_xticks(xticks)
            axs[2].set_yticks(yticks)
            axs[2].set_xticklabels(x_label_list)
            axs[2].set_yticklabels(y_label_list)
        else:
            img_flux.set_data(flux[t])
            img_sst.set_data(sst[t])
            img_press.set_data(press[t])

        fig.colorbar(img_flux, cax=cax_flux, orientation='vertical')
        fig.colorbar(img_sst, cax=cax_sst, orientation='vertical')
        fig.colorbar(img_press, cax=cax_press, orientation='vertical')

        fig.tight_layout()
        plt.tight_layout()
        # fig.savefig(files_path_prefix + f'videos/3D/flux-sst-press/hourly/{pic_num:05d}.png')
        fig.savefig(files_path_prefix + f'videos/3D/flux-sst-press/correlations/{frequency}/{pic_num:05d}.png')
        pic_num += 1
    return
