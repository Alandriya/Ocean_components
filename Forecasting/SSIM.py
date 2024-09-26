from scipy import signal
from scipy.ndimage.filters import convolve
from config import cfg
import numpy as np


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(cfg.data_type)
    img2 = img2.astype(cfg.data_type)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)), axis=(1, 2, 3))
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs

def get_SSIM(prediction, truth):
    """Calculate the SSIM score following
    [TIP2004] Image Quality Assessment: From Error Visibility to Structural Similarity
    Same functionality as
    https://github.com/coupriec/VideoPredictionICLR2016/blob/master/image_error_measures.lua#L50-L75
    We use nowcasting.helpers.msssim, which is borrowed from Tensorflow to do the evaluation
    Parameters
    ----------
    prediction : np.ndarray  sbchw
    truth : np.ndarray
    Returns
    -------
    ret : np.ndarray
    :param truth:
    :param prediction:
    """
    s, b, c, h, w = prediction.shape
    prediction = prediction.reshape((s * b, h, w, c))
    truth = truth.reshape((s * b, h, w, c))
    # if 'human3.6m' in cfg.GLOBAL.dataset:
    #     for i in range(s * b):
    #         prediction[i] = cv2.cvtColor(prediction[i], cv2.COLOR_BGR2GRAY)
    #         truth[i] = cv2.cvtColor(truth[i], cv2.COLOR_BGR2GRAY)
    #     prediction = prediction[:, np.newaxis]
    #     truth = truth[:, np.newaxis]

    ssim, cs = _SSIMForMultiScale(img1=prediction, img2=truth, max_val=1.0)
    ret = ssim.reshape((s, b))
    return ret