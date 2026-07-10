import numpy as np
import tqdm

from Data_processing.data_processing import scale_to_bins
from Data_processing.func_estimation import log_b2
from config import count_constants, sensible_params, latent_params

params = sensible_params
# params = latent_params
x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = params
dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(params)

a_coefs = k
b_args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]


def a(x):
    tmp = np.zeros_like(x)
    for i in range(5):
        tmp += a_coefs[i] * x**i
    return tmp

def b(x):
    tmp = [log_b2(x_i, b_args) for x_i in x]
    return np.sqrt(np.exp(np.array(tmp)))

def predict_step(amount: int,
                  x_start):
    prev = np.full(amount, x_start)
    eps = np.random.normal(0, 1, amount)
    ensemble = prev + a(prev) + b(prev) * eps
    return ensemble

def make_prediction(x_array: np.ndarray,
                    t_start: int,
                    t_end: int,
                    mask: np.ndarray,
                    n_bins: int,
                    n_steps: int,
                    amount_ensemble: int,
                    ):
    x_array_grouped, quantiles = scale_to_bins(x_array, n_bins)
    prediction = np.zeros((t_end - t_start, mask.shape[0], mask.shape[1]))
    prediction_q05 = np.zeros((t_end - t_start, mask.shape[0], mask.shape[1]))
    prediction_q95 = np.zeros((t_end - t_start, mask.shape[0], mask.shape[1]))
    prediction[:, np.logical_not(mask)] = np.nan
    prediction_q05[:, np.logical_not(mask)] = np.nan
    prediction_q95[:, np.logical_not(mask)] = np.nan
    for t in range(t_start, t_end, n_steps):
        print(f'Predicting time step {t} for {n_steps} steps')
        input = x_array[t-1]
        for t_pred in range(n_steps):
            for g in range(len(quantiles)-1):
                points = np.where((quantiles[g] <= input) & (input < quantiles[g + 1]))
                ensemble = predict_step(amount_ensemble, np.nanmean(input[points]))
                prediction[t - t_start + t_pred][points] = np.mean(ensemble)
                prediction_q05[t-t_start + t_pred][points] = np.quantile(ensemble, 0.05)
                prediction_q95[t - t_start + t_pred][points] = np.quantile(ensemble, 0.95)

            input = prediction[t-t_start + t_pred]

    return prediction, prediction_q05, prediction_q95


def count_crps():
    return