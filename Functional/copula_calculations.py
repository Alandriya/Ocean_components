import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kendalltau, norm, rankdata
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import t
from scipy.stats import beta

def normalize_1d_density(x, density):
    x = np.asarray(x, dtype=float)
    density = np.maximum(np.asarray(density, dtype=float), 0.0)
    normalization = np.trapezoid(density, x)
    if normalization <= 0:
        raise RuntimeError("Density normalization is zero.")
    return density / normalization


def calculate_cdf_from_density(x, density):
    density = normalize_1d_density(x, density)
    cdf = cumulative_trapezoid(density, x, initial=0.0)
    return cdf / cdf[-1]


def estimate_kendall_dependence(data1_array, data2_array, sample_size=300000, random_seed=12345):
    data1_array = np.asarray(data1_array)
    data2_array = np.asarray(data2_array)
    if data1_array.shape != data2_array.shape:
        raise ValueError("data1_array and data2_array must have the same shape.")

    rng = np.random.default_rng(random_seed)
    data1_flat = data1_array.ravel()
    data2_flat = data2_array.ravel()
    total_size = len(data1_flat)
    sampled_x1, sampled_x2 = [], []
    collected = 0

    while collected < sample_size:
        needed = sample_size - collected
        batch_size = max(2 * needed, 10000)
        indices = rng.integers(0, total_size, size=batch_size)
        x1 = data1_flat[indices]
        x2 = data2_flat[indices]
        valid = np.isfinite(x1) & np.isfinite(x2)
        x1 = x1[valid]
        x2 = x2[valid]
        take = min(needed, len(x1))
        sampled_x1.append(x1[:take])
        sampled_x2.append(x2[:take])
        collected += take

    sampled_x1 = np.concatenate(sampled_x1)
    sampled_x2 = np.concatenate(sampled_x2)
    tau, p_value = kendalltau(sampled_x1, sampled_x2)
    rho = np.sin(0.5 * np.pi * tau)

    print(f"Kendall tau: {tau}")
    print(f"Gaussian copula rho: {rho}")
    print(f"p-value: {p_value}")
    print(f"sample size: {len(sampled_x1)}")

    return {"tau": tau, "rho": rho, "p_value": p_value, "sampled_x1": sampled_x1, "sampled_x2": sampled_x2}


def gaussian_copula_density(u, v, rho, epsilon=1e-10):
    u = np.clip(u, epsilon, 1.0 - epsilon)
    v = np.clip(v, epsilon, 1.0 - epsilon)
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)
    denominator = 1.0 - rho ** 2
    exponent = (2.0 * rho * z1 * z2 - rho ** 2 * (z1 ** 2 + z2 ** 2)) / (2.0 * denominator)
    return np.exp(exponent) / np.sqrt(denominator)


def build_gaussian_copula_stationary_density(x1, p1, x2, p2, rho):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)
    cdf1 = calculate_cdf_from_density(x1, p1)
    cdf2 = calculate_cdf_from_density(x2, p2)
    u, v = np.meshgrid(cdf1, cdf2)
    copula_density = gaussian_copula_density(u, v, rho)
    p1_mesh, p2_mesh = np.meshgrid(p1, p2)
    joint_density = copula_density * p1_mesh * p2_mesh
    normalization = np.trapezoid(np.trapezoid(joint_density, x1, axis=1), x2)
    joint_density /= normalization
    return {"joint_density": joint_density, "p1": p1, "p2": p2, "cdf1": cdf1, "cdf2": cdf2, "rho": rho}


def calculate_copula_marginals(joint_density, x1, x2):
    marginal_x1 = np.trapezoid(joint_density, x2, axis=0)
    marginal_x2 = np.trapezoid(joint_density, x1, axis=1)
    return marginal_x1, marginal_x2


def calculate_joint_density_moments(joint_density, x1, x2):
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    mean_x1 = np.trapezoid(np.trapezoid(x1_mesh * joint_density, x1, axis=1), x2)
    mean_x2 = np.trapezoid(np.trapezoid(x2_mesh * joint_density, x1, axis=1), x2)
    var_x1 = np.trapezoid(np.trapezoid((x1_mesh - mean_x1) ** 2 * joint_density, x1, axis=1), x2)
    var_x2 = np.trapezoid(np.trapezoid((x2_mesh - mean_x2) ** 2 * joint_density, x1, axis=1), x2)
    covariance = np.trapezoid(np.trapezoid((x1_mesh - mean_x1) * (x2_mesh - mean_x2) * joint_density, x1, axis=1), x2)
    correlation = covariance / np.sqrt(var_x1 * var_x2)
    return {"mean_x1": mean_x1, "mean_x2": mean_x2, "sd_x1": np.sqrt(var_x1), "sd_x2": np.sqrt(var_x2), "covariance": covariance, "correlation": correlation}


def calculate_marginal_errors(joint_density, x1, p1, x2, p2):
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)
    marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1, x2)
    error_x1 = np.trapezoid(np.abs(marginal_x1 - p1), x1)
    error_x2 = np.trapezoid(np.abs(marginal_x2 - p2), x2)
    return error_x1, error_x2


def print_copula_summary(dependence, moments, error_x1, error_x2):
    print(f"Kendall tau: {dependence['tau']}")
    print(f"Gaussian copula rho: {dependence['rho']}")
    print(f"Resulting Pearson correlation: {moments['correlation']}")
    print(f"L1 marginal error X1: {error_x1}")
    print(f"L1 marginal error X2: {error_x2}")

def build_empirical_copula_density(sampled_x1, sampled_x2, bins=60):
    sampled_x1 = np.asarray(sampled_x1, dtype=float)
    sampled_x2 = np.asarray(sampled_x2, dtype=float)
    valid = np.isfinite(sampled_x1) & np.isfinite(sampled_x2)
    sampled_x1 = sampled_x1[valid]
    sampled_x2 = sampled_x2[valid]

    n = len(sampled_x1)
    u = rankdata(sampled_x1, method="average") / (n + 1.0)
    v = rankdata(sampled_x2, method="average") / (n + 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    density, _, _ = np.histogram2d(v, u, bins=[edges, edges], density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return {"u": u, "v": v, "density": density, "centers": centers, "edges": edges}


def build_gaussian_copula_density_grid(rho, centers):
    u_mesh, v_mesh = np.meshgrid(centers, centers)
    density = gaussian_copula_density(u_mesh, v_mesh, rho)

    du = centers[1] - centers[0]
    density /= np.sum(density) * du ** 2

    return density


def compare_copula_densities(empirical_density, gaussian_density, centers):
    du = centers[1] - centers[0]
    empirical_probability = empirical_density * du ** 2
    gaussian_probability = gaussian_density * du ** 2

    empirical_probability /= empirical_probability.sum()
    gaussian_probability /= gaussian_probability.sum()

    tv = 0.5 * np.sum(np.abs(empirical_probability - gaussian_probability))
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(empirical_probability) - np.sqrt(gaussian_probability)) ** 2))

    return {"total_variation": tv, "hellinger": hellinger}

def student_t_copula_density(u, v, rho, df, epsilon=1e-8):
    u = np.clip(u, epsilon, 1.0 - epsilon)
    v = np.clip(v, epsilon, 1.0 - epsilon)

    z1 = t.ppf(u, df)
    z2 = t.ppf(v, df)
    denominator = 1.0 - rho ** 2

    log_joint = gammaln((df + 2.0) / 2.0) - gammaln(df / 2.0)
    log_joint -= np.log(df * np.pi) + 0.5 * np.log(denominator)
    log_joint -= (df + 2.0) / 2.0 * np.log1p((z1 ** 2 - 2.0 * rho * z1 * z2 + z2 ** 2) / (df * denominator))

    log_marginal1 = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
    log_marginal1 -= 0.5 * np.log(df * np.pi)
    log_marginal1 -= (df + 1.0) / 2.0 * np.log1p(z1 ** 2 / df)

    log_marginal2 = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
    log_marginal2 -= 0.5 * np.log(df * np.pi)
    log_marginal2 -= (df + 1.0) / 2.0 * np.log1p(z2 ** 2 / df)

    return np.exp(log_joint - log_marginal1 - log_marginal2)


def fit_student_t_copula(u, v, rho, df_min=2.01, df_max=100.0):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.isfinite(u) & np.isfinite(v)
    u = u[valid]
    v = v[valid]

    def negative_log_likelihood(df):
        density = student_t_copula_density(u, v, rho, df)
        return -np.sum(np.log(np.maximum(density, 1e-300)))

    result = minimize_scalar(negative_log_likelihood, bounds=(df_min, df_max), method="bounded")

    if not result.success:
        raise RuntimeError("Student t-copula optimization failed.")

    print(f"Student t-copula rho: {rho}")
    print(f"Student t-copula df: {result.x}")
    print(f"Negative log-likelihood: {result.fun}")

    return {"rho": rho, "df": result.x, "negative_log_likelihood": result.fun}


def build_student_t_copula_density_grid(rho, df, centers):
    u_mesh, v_mesh = np.meshgrid(centers, centers)
    density = student_t_copula_density(u_mesh, v_mesh, rho, df)
    du = centers[1] - centers[0]
    density /= np.sum(density) * du ** 2
    return density


def calculate_student_t_tail_dependence(rho, df):
    value = 2.0 * t.cdf(-np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho)), df + 1.0)
    return value

def build_checkerboard_copula(u, v, bins=50, tolerance=1e-12, maximum_iterations=10000):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.isfinite(u) & np.isfinite(v)
    u = np.clip(u[valid], 0.0, 1.0)
    v = np.clip(v[valid], 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _, _ = np.histogram2d(v, u, bins=[edges, edges])

    probability_raw = counts / counts.sum()
    probability = probability_raw.copy()
    target = np.full(bins, 1.0 / bins)

    if np.any(probability.sum(axis=0) == 0) or np.any(probability.sum(axis=1) == 0):
        raise RuntimeError("Checkerboard contains an empty marginal bin.")

    error = np.inf

    for iteration in range(maximum_iterations):
        probability *= (target / probability.sum(axis=1))[:, None]
        probability *= (target / probability.sum(axis=0))[None, :]

        row_error = np.max(np.abs(probability.sum(axis=1) - target))
        column_error = np.max(np.abs(probability.sum(axis=0) - target))
        error = max(row_error, column_error)

        if error < tolerance:
            break

    density = probability * bins ** 2
    raw_density = probability_raw * bins ** 2
    correction_tv = 0.5 * np.sum(np.abs(probability - probability_raw))

    print(f"Checkerboard bins: {bins} x {bins}")
    print(f"Checkerboard iterations: {iteration + 1}")
    print(f"Checkerboard marginal error: {error}")
    print(f"TV introduced by marginal balancing: {correction_tv}")

    return {
        "density": density,
        "raw_density": raw_density,
        "probability": probability,
        "centers": centers,
        "edges": edges,
        "bins": bins,
        "balancing_error": error,
        "balancing_tv": correction_tv,
    }


def trapezoid_weights(x):
    x = np.asarray(x, dtype=float)
    weights = np.empty(len(x))
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def balance_joint_density_marginals(joint_density, x1, p1, x2, p2, tolerance=1e-12, maximum_iterations=10000):
    w1 = trapezoid_weights(x1)
    w2 = trapezoid_weights(x2)

    target1 = p1 * w1
    target2 = p2 * w2
    target1 /= target1.sum()
    target2 /= target2.sum()

    joint_probability = joint_density * w2[:, None] * w1[None, :]
    joint_probability /= joint_probability.sum()

    error = np.inf

    for iteration in range(maximum_iterations):
        row_sum = joint_probability.sum(axis=1)
        row_factor = np.divide(target2, row_sum, out=np.ones_like(target2), where=row_sum > 0)
        joint_probability *= row_factor[:, None]

        column_sum = joint_probability.sum(axis=0)
        column_factor = np.divide(target1, column_sum, out=np.ones_like(target1), where=column_sum > 0)
        joint_probability *= column_factor[None, :]

        row_error = np.max(np.abs(joint_probability.sum(axis=1) - target2))
        column_error = np.max(np.abs(joint_probability.sum(axis=0) - target1))
        error = max(row_error, column_error)

        if error < tolerance:
            break

    joint_density = joint_probability / (w2[:, None] * w1[None, :])

    print(f"Physical marginal balancing iterations: {iteration + 1}")
    print(f"Physical marginal balancing error: {error}")

    return joint_density


def build_checkerboard_stationary_density(x1, p1, x2, p2, checkerboard, enforce_marginals=True):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)

    cdf1 = calculate_cdf_from_density(x1, p1)
    cdf2 = calculate_cdf_from_density(x2, p2)

    edges = checkerboard["edges"]
    copula_density = checkerboard["density"]
    bins = checkerboard["bins"]

    index1 = np.searchsorted(edges, cdf1, side="right") - 1
    index2 = np.searchsorted(edges, cdf2, side="right") - 1
    index1 = np.clip(index1, 0, bins - 1)
    index2 = np.clip(index2, 0, bins - 1)

    copula_on_grid = copula_density[np.ix_(index2, index1)]
    joint_density = copula_on_grid * p2[:, None] * p1[None, :]

    normalization = np.trapezoid(np.trapezoid(joint_density, x1, axis=1), x2)
    joint_density /= normalization

    if enforce_marginals:
        joint_density = balance_joint_density_marginals(joint_density, x1, p1, x2, p2)

    marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1, x2)
    error_x1 = np.trapezoid(np.abs(marginal_x1 - p1), x1)
    error_x2 = np.trapezoid(np.abs(marginal_x2 - p2), x2)
    moments = calculate_joint_density_moments(joint_density, x1, x2)

    print(f"Checkerboard stationary L1 marginal error X1: {error_x1}")
    print(f"Checkerboard stationary L1 marginal error X2: {error_x2}")
    print(f"Checkerboard stationary Pearson correlation: {moments['correlation']}")
    print(f"Checkerboard stationary covariance: {moments['covariance']}")

    return {
        "joint_density": joint_density,
        "copula_on_grid": copula_on_grid,
        "marginal_x1": marginal_x1,
        "marginal_x2": marginal_x2,
        "p1": p1,
        "p2": p2,
        "cdf1": cdf1,
        "cdf2": cdf2,
        "moments": moments,
        "error_x1": error_x1,
        "error_x2": error_x2,
    }

def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_empirical_2d_density(data1_array, data2_array, x1, x2, chunk_size=2_000_000):
    x_edges = centers_to_edges(x1)
    y_edges = centers_to_edges(x2)
    counts = np.zeros((len(x2), len(x1)), dtype=float)

    data1_flat = np.asarray(data1_array).ravel()
    data2_flat = np.asarray(data2_array).ravel()

    if data1_flat.shape != data2_flat.shape:
        raise ValueError("data1_array and data2_array must have the same shape.")

    for start in range(0, len(data1_flat), chunk_size):
        end = min(start + chunk_size, len(data1_flat))
        x1_chunk = data1_flat[start:end]
        x2_chunk = data2_flat[start:end]
        valid = np.isfinite(x1_chunk) & np.isfinite(x2_chunk)

        chunk_counts, _, _ = np.histogram2d(
            x2_chunk[valid],
            x1_chunk[valid],
            bins=[y_edges, x_edges],
        )
        counts += chunk_counts

    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    area = dy[:, None] * dx[None, :]

    retained = counts.sum()
    density = counts / retained / area

    print(f"Empirical 2D observations inside grid: {int(retained)}")
    print(f"Empirical 2D density normalization: {np.sum(density * area)}")

    return density

def evaluate_bernstein_copula(u, v, probability):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    probability = np.asarray(probability, dtype=float)

    m = probability.shape[0]
    if probability.shape != (m, m):
        raise ValueError("Probability matrix must be square.")

    indices = np.arange(1, m + 1)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    v = np.clip(v, 1e-10, 1.0 - 1e-10)

    basis_u = beta.pdf(u[None, :], indices[:, None], (m + 1 - indices)[:, None])
    basis_v = beta.pdf(v[None, :], indices[:, None], (m + 1 - indices)[:, None])

    return basis_v.T @ probability @ basis_u


def build_bernstein_copula_grid(checkerboard, grid_size=200):
    probability = checkerboard["probability"]
    grid = np.linspace(0.005, 0.995, grid_size)
    density = evaluate_bernstein_copula(grid, grid, probability)

    du = grid[1] - grid[0]
    normalization = np.sum(density) * du ** 2

    print(f"Bernstein copula degree: {probability.shape[0]}")
    print(f"Bernstein grid normalization: {normalization}")

    return {"grid": grid, "density": density, "probability": probability}


def build_bernstein_stationary_density(x1, p1, x2, p2, checkerboard):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)

    cdf1 = calculate_cdf_from_density(x1, p1)
    cdf2 = calculate_cdf_from_density(x2, p2)

    copula_density = evaluate_bernstein_copula(cdf1, cdf2, checkerboard["probability"])
    joint_density = copula_density * p2[:, None] * p1[None, :]

    normalization = np.trapezoid(np.trapezoid(joint_density, x1, axis=1), x2)
    joint_density /= normalization

    marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1, x2)
    error_x1 = np.trapezoid(np.abs(marginal_x1 - p1), x1)
    error_x2 = np.trapezoid(np.abs(marginal_x2 - p2), x2)
    moments = calculate_joint_density_moments(joint_density, x1, x2)

    print(f"Bernstein stationary normalization: {normalization}")
    print(f"Bernstein L1 marginal error X1: {error_x1}")
    print(f"Bernstein L1 marginal error X2: {error_x2}")
    print(f"Bernstein Pearson correlation: {moments['correlation']}")
    print(f"Bernstein covariance: {moments['covariance']}")

    return {
        "joint_density": joint_density,
        "copula_density": copula_density,
        "marginal_x1": marginal_x1,
        "marginal_x2": marginal_x2,
        "p1": p1,
        "p2": p2,
        "cdf1": cdf1,
        "cdf2": cdf2,
        "moments": moments,
        "error_x1": error_x1,
        "error_x2": error_x2,
    }