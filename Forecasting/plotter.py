import matplotlib.pyplot as plt
import numpy as np
import tqdm

from video import get_continuous_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

files_path_prefix = 'D://Data/OceanFull/'


def plot_predictions(files_path_prefix: str,
                     Y_test: np.ndarray,
                     Y_predict: np.ndarray,
                     model_name: str):
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    axs[0].set_title('Real values', fontsize=20)
    axs[1].set_title('Predicted values', fontsize=20)
    axs[2].set_title('Absolute difference', fontsize=20)

    img = [None for _ in range(3)]
    cax = [None for _ in range(3)]

    y_min = min(np.nanmin(Y_test), np.nanmin(Y_predict))
    y_max = max(np.nanmax(Y_test), np.nanmax(Y_predict))
    # cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - y_min) / (y_max - y_min), 1])
    cmap = plt.get_cmap('Blues')
    cmap.set_bad('darkgreen', 1.0)
    for t in tqdm.tqdm(range(Y_test.shape[0])):
        difference = np.abs(Y_predict[t] - Y_test[t])
        rmse = np.sqrt(np.sum(np.square(difference)))
        fig.suptitle(f'{model_name}, day {t}, RMSE = {rmse:.2f}', fontsize=30)

        for i in range(3):
            divider = make_axes_locatable(axs[i])
            cax[i] = divider.append_axes('right', size='5%', pad=0.3)
        if img[0] is None:
            img[0] = axs[0].imshow(Y_test[t],
                                   interpolation='none',
                                   cmap=cmap,
                                   vmin=y_min,
                                   vmax=y_max)

            img[1] = axs[1].imshow(Y_predict[t],
                                   interpolation='none',
                                   cmap=cmap,
                                   vmin=y_min,
                                   vmax=y_max)

            img[2] = axs[2].imshow(difference,
                                   interpolation='none',
                                   cmap=plt.get_cmap('Reds'),
                                   vmin=0,
                                   vmax=np.nanmax(difference))
        else:
            img[0].set_data(Y_test[t])
            img[1].set_data(Y_predict[t])
            img[2].set_data(difference)

        for i in range(3):
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')

        plt.tight_layout()
        fig.savefig(files_path_prefix + f'videos/Forecast/{model_name}_{t}.png')
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
