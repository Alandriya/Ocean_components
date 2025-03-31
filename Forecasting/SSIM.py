# from numpy.ma.core import max_val, min_val
import numpy as np
from skimage.metrics import structural_similarity as ssim

from Forecasting.config import cfg


def get_SSIM(prediction, truth, mask=None):
    b, s, c, h, w = prediction.shape
    ssim_arr = np.zeros((b, s, 3))

    if not np.isnan(mask).any():
        prediction_flat = np.reshape(prediction, (b, s, c, h * w))
        prediction_flat = np.take(prediction_flat, np.where(mask.flatten())[0], axis=3)
        truth_flat = np.reshape(truth, (b, s, c, h * w))
        truth_flat = np.take(truth_flat, np.where(mask.flatten())[0], axis=3)
        # print(prediction_flat[0, :, 0, 0])
        # print(truth_flat[0, :, 0, 0])

        max_pred = np.max(prediction_flat, axis=(0, 1, 3))
        min_pred = np.min(prediction_flat, axis=(0, 1, 3))
        max_true = np.max(truth_flat, axis=(0, 1, 3))
        min_true = np.min(truth_flat, axis=(0, 1, 3))
        # print(max_pred)
        # print(min_pred)
        # print(max_true)
        # print(min_true)
        # print('\n\n\n')

        for i in range(b):
            for day in range(s):
                for k in range(cfg.channels):
                    # ssim_arr[i, day, k] = ssim(prediction[i, day, k], truth[i, day, k], data_range=1)
                    max_val = max(max_pred[k], max_true[k])
                    min_val = min(min_pred[k], min_true[k])
                    ssim_arr[i, day, k] = ssim(prediction_flat[i, day, k], truth_flat[i, day, k], data_range=max_val - min_val)
        return ssim_arr

    for i in range(b):
        for day in range(s):
            for k in range(3):
                # ssim_arr[i, day, k] = ssim(prediction[i, day, k], truth[i, day, k], data_range=1)
                ssim_arr[i, day, k] = ssim(prediction[i, day, k], truth[i, day, k], data_range=1)
    return ssim_arr