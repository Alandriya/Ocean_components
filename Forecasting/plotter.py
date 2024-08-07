import datetime
import os
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
files_path_prefix = 'D://Data/OceanFull/'


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
                     start_day: datetime.datetime,
                     mask: np.ndarray):
    print('Plotting')
    if not os.path.exists(files_path_prefix + f'videos/Forecast/{model_name}'):
        os.mkdir(files_path_prefix + f'videos/Forecast/{model_name}')

    days_prediction = Y_predict.shape[2]


    # axs[0].set_title('Real values', fontsize=20)
    # axs[1].set_title('Predicted values', fontsize=20)
    # axs[2].set_title('Absolute difference', fontsize=20)

    for k in range(3):
        fig, axs = plt.subplots(3, days_prediction, figsize=(5 * days_prediction, 15))

        img = [[None for _ in range(days_prediction)] for _ in range(3)]
        cax = [[None for _ in range(days_prediction)] for _ in range(3)]

        y_min = np.nanmin(Y_test[:, :, :, k]) / 2
        y_max = np.nanmax(Y_test[:, :, :, k]) / 2

        test_min = np.nanmin(Y_test[:, :, :, k])
        test_max = np.nanmax(Y_test[:, :, :, k])

        mape = mean_absolute_percentage_error(((Y_test[:, :, :, k]-test_min)/(test_max - test_min)).flatten(),
                                              ((Y_predict[:, :, :, k]-test_min)/(test_max - test_min)).flatten())

        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - y_min) / (y_max - y_min), 1])
        cmap.set_bad('darkgreen', 1.0)

        cmap_diff = plt.get_cmap('Reds').copy()
        cmap_diff.set_bad('darkgreen', 1.0)

        for t in tqdm.tqdm(range(days_prediction)):
            Y_test[np.logical_not(mask), :, k] = np.nan
            Y_predict[np.logical_not(mask), :, k] = np.nan
            difference = np.array(np.abs(Y_predict[:, :, t, k] - Y_test[:, :, t, k]))

            day_str = (start_day + datetime.timedelta(days=t)).strftime('%d.%m.%Y')
            for i in range(3):
                divider = make_axes_locatable(axs[i][t])
                cax[i][t] = divider.append_axes('right', size='5%', pad=0.3)

                axs[0][t].set_title(f'{day_str}, real values', fontsize=16)
                img[0][t] = axs[0][t].imshow(Y_test[:, :, t, k],
                                             interpolation='none',
                                             cmap=cmap,
                                             vmin=y_min,
                                             vmax=y_max)

                axs[1][t].set_title(f'{day_str}, predictions', fontsize=16)
                img[1][t] = axs[1][t].imshow(Y_predict[:, :, t, k],
                                             interpolation='none',
                                             cmap=cmap,
                                             vmin=y_min,
                                             vmax=y_max)

                axs[2][t].set_title(f'{day_str}, absolute difference', fontsize=16)
                img[2][t] = axs[2][t].imshow(difference,
                                             interpolation='none',
                                             cmap=cmap_diff,
                                             vmin=0,
                                             vmax=np.nanmax(difference))

            for i in range(3):
                fig.colorbar(img[i][t], cax=cax[i][t], orientation='vertical')

        if k == 0:
            fig.suptitle(f'{model_name}, Flux, MAPE with normalized test = {mape: .2e}', fontsize=30)
        elif k == 1:
            fig.suptitle(f'{model_name}, SST, MAPE with normalized test = {mape: .2e}', fontsize=30)
        else:
            fig.suptitle(f'{model_name}, Pressure, MAPE with normalized test = {mape: .2e}', fontsize=30)
        plt.tight_layout()
        if k == 0:
            fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{model_name}_Flux.png')
        elif k == 1:
            fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{model_name}_SST.png')
        else:
            fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}/{model_name}_press.png')
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


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]}')
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)
