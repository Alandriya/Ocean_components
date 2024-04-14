import datetime
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
                     start_day: datetime.datetime):
    print('Plotting')
    days_prediction = Y_predict.shape[2]
    fig, axs = plt.subplots(3, days_prediction, figsize=(5*days_prediction, 15))

    # axs[0].set_title('Real values', fontsize=20)
    # axs[1].set_title('Predicted values', fontsize=20)
    # axs[2].set_title('Absolute difference', fontsize=20)

    img = [[None for _ in range(days_prediction)] for _ in range(3)]
    cax = [[None for _ in range(days_prediction)] for _ in range(3)]

    y_min = min(np.nanmin(Y_test), np.nanmin(Y_predict))
    y_max = max(np.nanmax(Y_test), np.nanmax(Y_predict))
    cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - y_min) / (y_max - y_min), 1])
    # cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('darkgreen', 1.0)

    cmap_diff = plt.get_cmap('Reds').copy()
    cmap_diff.set_bad('darkgreen', 1.0)

    mse = 0
    for t in tqdm.tqdm(range(days_prediction)):
        difference = np.array(np.abs(Y_predict[:, :, t] - Y_test[:, :, t]))
        mse += np.sum(np.square(np.nan_to_num(difference)))

        day_str = (start_day + datetime.timedelta(days=t)).strftime('%d.%m.%Y')
        for i in range(3):
            divider = make_axes_locatable(axs[i][t])
            cax[i][t] = divider.append_axes('right', size='5%', pad=0.3)

            axs[0][t].set_title(f'{day_str}, real values', fontsize=16)
            img[0][t] = axs[0][t].imshow(Y_test[:, :, t],
                                         interpolation='none',
                                         cmap=cmap,
                                         vmin=y_min,
                                         vmax=y_max)

            axs[1][t].set_title(f'{day_str}, predictions', fontsize=16)
            img[1][t] = axs[1][t].imshow(Y_predict[:, :, t],
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

    fig.suptitle(f'{model_name}, RMSE = {np.sqrt(mse):.1f}', fontsize=30)
    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}.png')
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
