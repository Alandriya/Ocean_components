import datetime
import os
import pandas as pd
import tqdm
# from VarGamma import fit_ml, pdf, cdf
# from Plotting.plot_func_estimations import plot_estimate_a_flux
from Data_processing.data_processing import load_ABCFE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.cm
from scipy.special import gamma, gammaincc
from Data_processing.data_processing import scale_to_bins, mean_blocks
from scipy.integrate import trapezoid
# import warnings
# warnings.filterwarnings("error")

months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'}

font = {'size': 14}
matplotlib.rc('font', **font)


def estimate_a_flux_by_months(files_path_prefix: str, month: int, point, radius):
    """
    Estimates the dependence of A coefficient from flux values in shape of func, the estimation is carried on all data
    of fixed month: e.g, all Januaries, all Februaries, ...

    :param files_path_prefix: path to the working directory
    :param month: month number from 1 to 12
    :param point: the center of the square from which the data is used
    :param radius: the radius of the bigger point
    :return:
    """
    sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')

    biases = [i for i in range(-radius, radius+1)]
    point_bigger = [(point[0] + i, point[1] + j) for i in biases for j in biases]
    flat_points = np.array([p[0] * 181 + p[1] for p in point_bigger])

    df_sens = pd.DataFrame(columns=['dates', 'a', 'b', 'c', 'd', 'ss'])
    df_lat = pd.DataFrame(columns=['dates', 'a', 'b', 'c', 'd', 'ss'])

    if not os.path.exists(files_path_prefix + f"Func_repr/a-flux-monthly/{month}"):
        os.mkdir(files_path_prefix + f"Func_repr/a-flux-monthly/{month}")

    years = 10
    max_year = 1988

    sens_fits, lat_fits = list(), list()
    for i in range(0, years):
        time_start = (datetime.datetime(1979 + i, month, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
        if month != 12:
            time_end = (datetime.datetime(1979 + i + 2, month + 0, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
        else:
            time_end = (datetime.datetime(1979 + i + 1, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days

        a_timelist, _, _, _, _, borders = load_ABCFE(files_path_prefix, time_start + 1, time_end + 1, load_a=True)
        # sens_fit, lat_fit, sens_err, lat_err = plot_estimate_a_flux(files_path_prefix, a_timelist, borders,
        #                                                                   sensible_array, latent_array, time_start,
        #                                                                   time_end, month=month, flat_points=flat_points,
        #                                                                   point_center=point[0] * 181 + point[1])
        del a_timelist
        date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
        date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
        # sens_fits.append(sens_fit)
        # lat_fits.append(lat_fit)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    x = np.linspace(np.nanmin(sensible_array), np.nanmax(sensible_array), 100)
    for i in range(0, 10):
        # sens_params = df_sens[['a', 'b', 'c', 'd']].loc[i].values
        # lat_params = df_lat[['a', 'b', 'c', 'd']].loc[i].values
        if i > 30:
            color = plt.cm.tab20(i % 20)
        else:
            color = 'gray'
        axes[0].plot(x, sens_fits[i](x), label=f'{1979 + i}', c=color)
        axes[1].plot(x, lat_fits[i](x), label=f'{1979 + i}', c=color)

    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    axes[0].set_title('Sensible')
    axes[1].set_title('Latent')
    fig.suptitle(months_names[month])
    fig.tight_layout()
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{months_names[month]}.png")
    plt.close(fig)

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.bar(range(1979, max_year), df_sens['ss'])
    fig.suptitle('Sensible - sum of squared residuals')
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/{month}_error_sensible.png")
    plt.clf()

    axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.bar(range(1979, max_year), df_lat['ss'])
    fig.suptitle('Latent - sum of squared residuals')
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/{month}_error_latent.png")
    plt.close(fig)
    return

def estimate_A_B(files_path_prefix: str,
                 x: np.ndarray,
                 a:np.ndarray,
                 b:np.ndarray,
                 quantiles_amount: int = 250):
    x_grouped, _ = scale_to_bins(x, quantiles_amount)
    quantiles = np.unique(x_grouped)
    quantiles = quantiles[np.logical_not(np.isnan(quantiles))]
    quantiles = quantiles[quantiles != 0] #hotfix
    # quantiles = np.sort(quantiles)

    part = len(quantiles) // 20
    quantiles = quantiles[part:-part]

    a_grouped = np.zeros(len(quantiles))
    b_grouped = np.zeros(len(quantiles))
    a_full = list()
    b_full = list()
    x_full = list()

    print(len(quantiles))
    for g in tqdm.tqdm(range(len(quantiles))):
        quantile = quantiles[g]
        # data = x[x_grouped == quantile]
        if not np.isnan(quantile):
            # print(quantile)
            # print(np.sum((x_grouped==quantile)))
            a_grouped[g] = np.mean(a[x_grouped == quantile])
            b_grouped[g] = np.mean(b[x_grouped == quantile])
            a_full += list(a[x_grouped == quantile].flatten())
            b_full += list(b[x_grouped == quantile].flatten())
            x_full += list(x[x_grouped == quantile].flatten())

    return quantiles, a_grouped, b_grouped, x_full, a_full, b_full


def get_values(arr_grouped):
    quantiles = np.unique(arr_grouped)
    quantiles = quantiles[np.logical_not(np.isnan(quantiles))]
    quantiles = quantiles[quantiles != 0] #hotfix
    return quantiles


def kl_coeff(x, n, l):
    if n == 0:
        return 1 / l
    elif n == 1:
        return x / l - 1 / (l ** 2)
    elif n == 2:
        return x ** 2 / l - (2 * x) / (l ** 2) + 2 / (l ** 3)
    elif n == 3:
        return x ** 3 / l - (3 * x ** 2) / (l ** 2) + (6 * x) / (l ** 3) - 6 / (l ** 4)
    elif n == 4:
        return x ** 4 / l - (4 * x ** 3) / (l ** 2) + (12 * x ** 2) / (l ** 3) - (24 * x) / (l ** 4) + 24 / (l ** 5)
    else:
        raise ValueError("n must be 0..4")

def Phi(x, l, k):
    k0, k1, k2, k3, k4 = k
    return np.exp(l * x) * (
            k4 * kl_coeff(x, 4, l)
            + k3 * kl_coeff(x, 3, l)
            + k2 * kl_coeff(x, 2, l)
            + k1 * kl_coeff(x, 1, l)
            + k0 * kl_coeff(x, 0, l)
    )


def Psi_minus(x, c, k):
    tmp = 0.0
    try:
        for n in range(5):
            tmp += ((-1) ** n) * k[n] * c ** (-2 * n - 2) * gammaincc(2 * n + 2, c * np.sqrt(-x)) * gamma(2 * n + 2)
    except RuntimeWarning:
        print(x)
        tmp = 0
    return 2.0 * tmp


def Psi_plus(x, c, k):
    tmp = 0.0
    for n in range(5):
        tmp += k[n] * c ** (-2 * n - 2) * gammaincc(2 * n + 2, c * np.sqrt(x)) * gamma(2 * n + 2)
    return -2.0 * tmp


def I(x, args):
    c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4 = args
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    prefL = 2 * np.exp(-dL)
    prefC = 2 * np.exp(-dC)
    prefR = 2 * np.exp(-dR)

    if x0 < 0:
        if x < x1:
            return prefL * Phi(x, c1, k) + const1
        elif x1 <= x < x0:
            return prefC * Psi_minus(x, c2, k) + const2
        elif x0 <= x < 0:
            return prefR * Psi_minus(x, c4, k) + const3
        else:
            return prefR * Psi_plus(x, c4, k) + const4

    else:
        if x < x1:
            return prefL * Phi(x, c1, k) + const1
        elif x1 <= x < 0:
            return prefC * Psi_minus(x, c2, k) + const2
        elif 0 <= x < x0:
            return prefC * Psi_plus(x, c2, k) + const3
        else:
            return prefR * Psi_plus(x, c4, k) + const4

def log_b2(x, args):
    c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4 = args
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    if x < x1:
        return c1 * abs(x) + dL
    elif x1 <= x < x0:
        return c2 * np.sqrt(abs(x)) + dC
    else:
        return c4 * np.sqrt(abs(x)) + dR


def create_quantiles_2d(files_path_prefix: str,
                        data1_array: np.ndarray,
                        data2_array: np.ndarray,
                        data1_name: str,
                        data2_name: str,
                        mask: np.ndarray,
                        quantiles_amount:int,
                        coef_start: int,
                        coef_end: int,
                        start_year: int,
                        block_size: int,
                        b_type:str):
    str_types = f'{data1_name}-{data2_name}'
    a_array = np.load(files_path_prefix + f'Components/{str_types}/A_{coef_start}-{coef_end}.npy')

    if b_type == 'eigen':
        b_array = np.load(files_path_prefix + f'Eigenvalues/{str_types}/B2_{coef_start}-{coef_end}_1d.npy')
        b_array = np.nan_to_num(b_array)
        b_array = np.swapaxes(b_array, 1, 3)
        b_array = np.swapaxes(b_array, 1, 2)
        if coef_start == 1 and coef_end == 17106:
            missing_days_eigen = [3652, 14609]
            a_array = np.delete(a_array, missing_days_eigen, axis=0)
    else:
        b_array = np.load(files_path_prefix + f'Components/{str_types}/B_{coef_start}-{coef_end}.npy')
        b_array = b_array**2

    print(f'Loaded data')

    data1_array[:, np.logical_not(mask)] = np.nan
    data2_array[:, np.logical_not(mask)] = np.nan
    a_array[:, np.logical_not(mask)] = np.nan
    b_array[:, np.logical_not(mask)] = np.nan

    start = 0
    end = np.min([a_array.shape[0], data1_array.shape[0], b_array.shape[0]])
    if start_year == 2019:
        end = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days

    data1_small = mean_blocks(data1_array[start:end] - a_array[start:end, :, :, 0], mask, block_size)
    data2_small = mean_blocks(data2_array[start:end] - a_array[start:end, :, :, 1], mask, block_size)
    print('Created blocks')

    a_small = mean_blocks(a_array[start:end], mask, block_size)
    b_small = mean_blocks(b_array[start:end], mask, block_size)

    # quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full, = (
    #     estimate_A_B_2d(data1_small, data2_small, a_small, b_small, quantiles_amount=260))
    x1_grouped, _ = scale_to_bins(data1_small, quantiles_amount)
    x2_grouped, _ = scale_to_bins(data2_small, quantiles_amount)
    quantiles1 = get_values(x1_grouped)
    quantiles2 = get_values(x2_grouped)

    quantiles1 = quantiles1[5:-5]
    quantiles2 = quantiles2[5:-5]

    a_grouped = np.zeros((2, len(quantiles1)))
    b_grouped = np.zeros((2, len(quantiles1)))

    for q1 in range(len(quantiles1)):
        quantile1 = quantiles1[q1]
        a_grouped[0, q1] = np.mean(a_small[:, :, :, 0][x1_grouped == quantile1])
        b_grouped[0, q1] = np.mean(b_small[:, :, :, 0][x1_grouped == quantile1])

    for q2 in range(len(quantiles2)):
        quantile2 = quantiles2[q2]
        a_grouped[1, q2] = np.mean(a_small[:, :, :, 1][x2_grouped == quantile2])
        if b_small.shape[3] == 2:
            b_grouped[1, q2] = np.mean(b_small[:, :, :, 1][x2_grouped == quantile2])
        else:
            b_grouped[1, q2] = np.mean(b_small[:, :, :, 3][x2_grouped == quantile2])
    print(f'Got {len(quantiles1)} quantiles')

    if not os.path.exists(files_path_prefix + f'Functional/{str_types}'):
        os.mkdir(files_path_prefix + f'Functional/{str_types}')
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles1.npy', quantiles1)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles2.npy', quantiles2)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_grouped.npy', a_grouped)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_grouped_log_{b_type}.npy', np.log(b_grouped))
    return


def get_isolines(
    prob,
    x: np.ndarray,
    amount: int,
) -> np.ndarray:
    """
    Return vertical isolines x_1, ..., x_{amount-1} such that each vertical
    slice under y = prob(x) has equal area.

    The intervals are:

        [x_min, x_1],
        [x_1, x_2],
        ...
        [x_{amount-1}, x_max]

    Each interval has area total_area / amount.

    Parameters
    ----------
    prob:
        Density function. Must accept a NumPy array and return a NumPy array.
    x:
        1D grid of x-values.
    amount:
        Number of equal-area vertical slices.

    Returns
    -------
    np.ndarray
        Array of length amount - 1 containing the x-coordinates of isolines.
    """
    x = np.asarray(x, dtype=float)

    y = np.array([prob(t) for t in x], dtype=float)

    dx = np.diff(x)
    y0 = y[:-1]
    y1 = y[1:]

    # Trapezoidal area of each interval.
    segment_areas = 0.5 * (y0 + y1) * dx
    cumulative = np.concatenate([[0.0], np.cumsum(segment_areas)])

    total_area = cumulative[-1]

    targets = total_area * np.arange(1, amount) / amount
    isolines = []

    for target in targets:
        # Find segment containing the target cumulative area.
        i = np.searchsorted(cumulative, target, side="right") - 1
        i = min(max(i, 0), len(dx) - 1)

        area_before = cumulative[i]
        area_needed = target - area_before

        x_left = x[i]
        width = dx[i]
        left_height = y[i]
        right_height = y[i + 1]

        # Solve exactly inside this segment assuming linear interpolation
        # of prob(x) between grid points.
        #
        # Area from x_left to x_left + s:
        #
        # A(s) = left_height * s
        #        + (right_height - left_height) * s^2 / (2 * width)
        #
        # We need A(s) = area_needed.
        if segment_areas[i] == 0:
            # Flat zero-density region. This case is rare if target is valid,
            # but we handle it defensively.
            s = 0.0
        else:
            slope_term = (right_height - left_height) / (2.0 * width)

            if abs(slope_term) < 1e-15:
                # Approximately constant density on the segment.
                s = area_needed / left_height
            else:
                a = slope_term
                b = left_height
                c = -area_needed

                discriminant = b * b - 4.0 * a * c
                discriminant = max(discriminant, 0.0)

                root1 = (-b + np.sqrt(discriminant)) / (2.0 * a)
                root2 = (-b - np.sqrt(discriminant)) / (2.0 * a)

                # Choose the root inside the segment.
                candidates = [r for r in (root1, root2) if -1e-12 <= r <= width + 1e-12]

                if not candidates:
                    raise RuntimeError("Failed to locate isoline inside segment")

                s = candidates[0]
                s = min(max(s, 0.0), width)

        isolines.append(x_left + s)

    return np.asarray([np.nanmin(x)] + isolines + [np.nanmax(x)])


def moments(pdf, x_grid):
    p = np.array([pdf(x) for x in x_grid], dtype=float)
    p /= trapezoid(p, x_grid)
    mu = trapezoid(x_grid * p, x_grid)
    var = trapezoid((x_grid - mu)**2 * p, x_grid)
    return mu, var


def count_constants(params):
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = params
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    prefL = 2 * np.exp(-dL)
    prefC = 2 * np.exp(-dC)
    prefR = 2 * np.exp(-dR)

    const1 = -prefL * Phi(x1, c1, k)
    const2 = const1 + prefL * Phi(x1, c1, k) - prefC * Psi_minus(x1, c2, k)
    const3 = const2 + prefC * Psi_minus(x0, c2, k) - prefR * Psi_minus(x0, c4, k)
    const4 = const3 + prefR * Psi_minus(0, c4, k) - prefR * Psi_plus(0, c4, k)

    return [dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4]

def model_logb2(x, x1, x0, c1, c2, c3, c4):
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    if x < x1:
        return c1 * abs(x) + dL
    elif x < x0:
        return c2 * np.sqrt(abs(x)) + dC
    else:
        return c4 * np.sqrt(abs(x)) + dR\

def create_mesh(files_path_prefix: str,
               data1_array: np.ndarray,
               data2_array: np.ndarray,
               quantiles_amount: int,
                coef_start: int,
                coef_end: int,
                str_types: str,
               a_array: np.ndarray,
               b_array: np.ndarray,
               c_array: np.ndarray = None,

               ):

    x1_grouped, _ = scale_to_bins(data1_array, quantiles_amount)
    x2_grouped, _ = scale_to_bins(data2_array, quantiles_amount)
    quantiles1 = get_values(x1_grouped)
    quantiles2 = get_values(x2_grouped)

    a_mesh = np.zeros((2, len(quantiles1), len(quantiles2)))
    b_mesh = np.zeros((4, len(quantiles1), len(quantiles2)))
    c_mesh = np.zeros((3, len(quantiles1), len(quantiles2)))
    for q1 in tqdm.tqdm(range(len(quantiles1))):
        quantile1 = quantiles1[q1]
        for q2 in range(len(quantiles2)):
            quantile2 = quantiles2[q2]
            a_mesh[0, q1, q2] = np.mean(a_array[:, :, :, 0][(x1_grouped == quantile1) & (x2_grouped == quantile2)])
            a_mesh[1, q1, q2] = np.mean(a_array[:, :, :, 1][(x1_grouped == quantile1) & (x2_grouped == quantile2)])

            b_mesh[0, q1, q2] = np.mean(b_array[:, :, :, 0][(x1_grouped == quantile1) & (x2_grouped == quantile2)])
            b_mesh[1, q1, q2] = np.mean(b_array[:, :, :, 1][(x1_grouped == quantile1) & (x2_grouped == quantile2)])
            b_mesh[2, q1, q2] = np.mean(b_array[:, :, :, 2][(x1_grouped == quantile1) & (x2_grouped == quantile2)])
            b_mesh[3, q1, q2] = np.mean(b_array[:, :, :, 3][(x1_grouped == quantile1) & (x2_grouped == quantile2)])

            c_mesh[0, q1, q2] = np.mean(c_array[:, :, :, 0][(x1_grouped == quantile1) & (x2_grouped == quantile2)])
            c_mesh[1, q1, q2] = np.mean(c_array[:, :, :, 1][(x1_grouped == quantile1) & (x2_grouped == quantile2)])
            c_mesh[2, q1, q2] = np.mean(c_array[:, :, :, 2][(x1_grouped == quantile1) & (x2_grouped == quantile2)])

    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles1_{quantiles_amount}.npy', quantiles1)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles2_{quantiles_amount}.npy', quantiles2)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_mesh_{quantiles_amount}.npy', a_mesh)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_mesh_{quantiles_amount}.npy', b_mesh)
    np.save(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_c_mesh_{quantiles_amount}.npy', c_mesh)
    return

def count_correlation_BTT_mesh(c_mesh: np.ndarray,):
    corr_mesh = np.zeros((c_mesh.shape[1], c_mesh.shape[2]))
    corr_mesh = c_mesh[2] / (np.sqrt(c_mesh[0] * c_mesh[1]))
    return corr_mesh