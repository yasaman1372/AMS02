
import numpy as np
import scipy.special
from scipy.stats import chi2 as scp_chi2, norm as scp_gaussian, poisson as scp_poisson
from scipy.interpolate import interp1d, PchipInterpolator


def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def gaussian_cdf(x, mu, sigma):
    return (1 + scipy.special.erf((x - mu) / np.sqrt(2 * sigma**2))) / 2

def exp_modified_gaussian(x, mu, sigma, tau):
    l2 = 1 / (2 * tau)
    ls = sigma**2 / tau
    return l2 * np.exp(l2 * (2 * mu + ls - 2 * x)) * scipy.special.erfc((mu + ls - x) / (2**0.5 * sigma))

def exponential_distribution(x, tau):
    return np.exp(-x / tau) / tau

def exp_tailed_gaussian(x, mu, sigma, ncut, tau):
    factor = tau / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut * sigma / tau - ncut**2 / 2)
    y = gaussian(x, mu, sigma)
    tail_high = x > mu + ncut * sigma
    tail_low = x < mu - ncut * sigma
    y[tail_high] = factor * exponential_distribution(x[tail_high] - mu, 1 / tau)
    y[tail_low] = factor * exponential_distribution(mu - x[tail_low], 1 / tau)
    return y

def asymm_exp_tailed_gaussian(x, mu, sigma, ncut_high, tau_high, ncut_low, tau_low):
    factor_high = tau_high / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut_high * sigma / tau_high - ncut_high**2 / 2)
    factor_low = tau_low / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut_low * sigma / tau_low - ncut_low**2 / 2)
    y = gaussian(x, mu, sigma)
    tail_high = x > mu + ncut_high * sigma
    tail_low = x < mu - ncut_low * sigma
    y[tail_high] = factor_high * exponential_distribution(x[tail_high] - mu, 1 / tau_high)
    y[tail_low] = factor_low * exponential_distribution(mu - x[tail_low], 1 / tau_low)
    return y

def asymm_gaussian(x, mu, sigma, alpha):
    high = gaussian(x, mu, sigma)
    low = alpha * gaussian(x, mu, sigma * alpha)
    return 2 / (1 + alpha) * (high * (x >= mu) + low * (x < mu))

def landau(x, m, w):
    xs = (x - m) / w
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(xs + np.exp(-xs)) / 2)

def novosibirsk(x, mu, sigma, k):
    xi = np.sqrt(np.log(4))
    normed = (x - mu) / sigma
    arg = 1 - normed * k
    arg_sel = arg > 0
    arg2 = arg_sel * np.log(np.maximum(arg, 1e-7))
    width = (np.arcsinh(k * xi) / xi)**2
    return np.exp(-arg2**2 / (2 * width) - width / 2) * arg_sel


def king_pdf_1d(x, sigma, gamma):
    norm = np.sqrt(gamma / (2 * np.pi)) * (gamma - 1) * scipy.special.gamma(gamma - 0.5) / (sigma * scipy.special.gamma(gamma + 1))
    pdf = 1 / (2 * np.pi * sigma**2) * (1 - 1 / gamma) * (1 + (x / sigma)**2 / (2 * gamma))**(-gamma)
    return pdf / norm

def king_cdf_1d(x, sigma, gamma):
    # includes not analytically solvable integral
    return NotImplementedError

def calculate_chisq(data, model, errors, n_parameters):
    nonzero = errors > 0
    residuals = (data[nonzero] - model[nonzero]) / errors[nonzero]
    chisq = np.sum(residuals**2)
    dof = nonzero.sum() - n_parameters
    return chisq, dof, chisq / dof


def fermi_function(x):
    return 1 / (np.exp(-x) + 1)

def inverse_fermi_function(x):
    return -np.log(1 / x - 1)

def scaled_fermi_function(x, n, m, w, y0):
    return n * fermi_function((x - m) / w) + y0

def shifted_fermi_function(x, m, w):
    return fermi_function((x - m) / w)


def bethe_bloch(beta, charge, k1, k2, k3):
    gamma = 1 / (1 - beta**2)
    return charge**2 / beta**2 * (k1 * np.log(beta * gamma) - k2 * beta**2 + k3)

def bethe_bloch_pm(momentum, mass, charge, k1, k2, k3):
    energy = np.sqrt(momentum**2 + mass**2)
    beta = momentum / energy
    gamma = energy / mass
    return charge**2 / beta**2 * (k1 * np.log(beta * gamma) - k2 * beta**2 + k3)


def calculate_efficiency(passed, all):
    return (passed + 1) / (all + 2)

def calculate_efficiency_error(passed, all):
    k = passed
    n = all
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - (k + 1)**2 / (n + 2)**2)

def calculate_efficiency_and_error(passed, all):
    return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)


def calculate_efficiency_weighted(passed_values, failed_values):
    return passed_values / (passed_values + failed_values + 1e-100)

def calculate_efficiency_error_weighted(passed_values, failed_values, passed_squared_values, failed_squared_values):
    return np.sqrt(passed_squared_values * failed_values**2 + failed_squared_values * passed_values**2) / (passed_values + failed_values + 1e-100)**2

def calculate_efficiency_and_error_weighted(passed_values, failed_values, passed_squared_values, failed_squared_values):
    return calculate_efficiency_weighted(passed_values, failed_values), calculate_efficiency_error_weighted(passed_values, failed_values, passed_squared_values, failed_squared_values)

def calculate_efficiency_and_rejection(signal_histogram, background_histogram):
    assert signal_histogram.dimensions == 1 and background_histogram.dimensions == 1
    assert signal_histogram.binnings[0] == background_histogram.binnings[0]
    cut_binning = signal_histogram.binnings[0]
    bin_edges = cut_binning.edges
    signal_mean, signal_std, _ = hist_mean_and_std(signal_histogram)
    background_mean, background_std, _ = hist_mean_and_std(background_histogram)
    if signal_mean > background_mean:
        cut_values = bin_edges[1:]
        signal_values = signal_histogram.values
        background_values = background_histogram.values
    else:
        cut_values = bin_edges[:0:-1]
        signal_values = signal_histogram.values[::-1]
        background_values = background_histogram.values[::-1]
    signal_cumulative = signal_values[::-1].cumsum()[::-1]
    background_cumulative = background_values[::-1].cumsum()[::-1]
    signal_total = signal_cumulative[0]
    background_total = background_cumulative[0]
    signal_efficiency, signal_efficiency_error = calculate_efficiency_and_error(signal_cumulative, signal_total)
    background_efficiency, background_efficiency_error = calculate_efficiency_and_error(background_cumulative, background_total)
    background_rejection = 1 / background_efficiency
    background_efficiency_relative_error = background_efficiency_error / background_efficiency
    background_rejection_error = background_efficiency_relative_error * background_rejection
    return signal_efficiency, signal_efficiency_error, background_rejection, background_rejection_error, cut_values

def calculate_signal_and_background_efficiency(signal_histogram, background_histogram):
    assert signal_histogram.dimensions == 1 and background_histogram.dimensions == 1
    assert signal_histogram.binnings[0] == background_histogram.binnings[0]
    cut_binning = signal_histogram.binnings[0]
    bin_edges = cut_binning.edges
    signal_mean, signal_std, _ = hist_mean_and_std(signal_histogram)
    background_mean, background_std, _ = hist_mean_and_std(background_histogram)
    if signal_mean > background_mean:
        cut_values = bin_edges[:-1]
        signal_values = signal_histogram.values
        signal_squared_values = signal_histogram.squared_values
        background_values = background_histogram.values
        background_squared_values = background_histogram.squared_values
    else:
        cut_values = bin_edges[:0:-1]
        signal_values = signal_histogram.values[::-1]
        signal_squared_values = signal_histogram.squared_values[::-1]
        background_values = background_histogram.values[::-1]
        background_squared_values = background_histogram.squared_values[::-1]
    signal_passed_cumulative = signal_values[::-1].cumsum()[::-1]
    signal_passed_cumulative_squared = signal_squared_values[::-1].cumsum()[::-1]
    signal_failed_cumulative = signal_values.cumsum()
    signal_failed_cumulative_squared = signal_squared_values.cumsum()
    background_passed_cumulative = background_values[::-1].cumsum()[::-1]
    background_passed_cumulative_squared = background_squared_values[::-1].cumsum()[::-1]
    background_failed_cumulative = background_values.cumsum()
    background_failed_cumulative_squared = background_squared_values.cumsum()
    signal_efficiency, signal_efficiency_error = calculate_efficiency_and_error_weighted(signal_passed_cumulative, signal_failed_cumulative, signal_passed_cumulative_squared, signal_failed_cumulative_squared)
    background_efficiency, background_efficiency_error = calculate_efficiency_and_error_weighted(background_passed_cumulative, background_failed_cumulative, background_passed_cumulative_squared, background_failed_cumulative_squared)
    return signal_efficiency, signal_efficiency_error, background_efficiency, background_efficiency_error, cut_values


def calculate_cut_value_for_efficiency(signal_histogram, target_efficiency):
    assert signal_histogram.dimensions == 1
    cut_binning = signal_histogram.binnings[0]
    bin_edges = cut_binning.edges
    cut_values = bin_edges[:-1]
    signal_values = signal_histogram.values
    signal_cumulative = signal_values[::-1].cumsum()[::-1]
    signal_total = signal_cumulative[0]
    signal_efficiency, signal_efficiency_error = calculate_efficiency_and_error(signal_cumulative, signal_total)
    return np.interp(target_efficiency, signal_efficiency[::-1], cut_values[::-1])


def weighted_mean(values, errors):
    weights = 1 / errors**2
    return (values * weights).sum() / weights.sum()


def lafferty_whyatt(edges, gamma):
    ex = 1 - gamma
    rmin = edges[:-1]
    rmax = edges[1:]
    return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma)


def calculate_li_ma_significance(n_on, n_off, alpha):
    return (2 * n_on * np.log((1 + alpha) * n_on / (alpha * (n_on + n_off))) + 2 * n_off * np.log((1 + alpha) * n_off / (n_on + n_off)))**0.5


def row_mean_and_std(bin_centers, weights, axis=1):
    values = np.expand_dims(bin_centers, axis=1-axis)
    n = weights.sum(axis=axis)
    mean = (values * weights).sum(axis=axis) / n
    residuals = ((values - np.expand_dims(mean, axis=axis))**2 * weights).sum(axis=axis)
    std = np.sqrt(1 / (n - 1) * residuals)
    return mean, std, std / np.sqrt(n)

def weighted_row_mean_and_std(bin_centers, weights, squared_weights, axis=1):
    values = np.expand_dims(bin_centers, axis=1-axis)
    mean = (values * weights).sum(axis=axis) / weights.sum(axis=axis)
    n = weights.sum(axis=axis)
    effective_n = (weights**2 / (squared_weights + (weights == 0))).sum(axis=axis)
    residuals = ((values - np.expand_dims(mean, axis=axis))**2 * weights).sum(axis=axis)
    std = np.sqrt(1 / (effective_n - 1) * residuals) / (n / effective_n)**0.5
    sum_of_weights_squared = weights.sum(axis=axis)**2
    sum_of_squared_weights = squared_weights.sum(axis=axis)
    mean_error = std / (effective_n * np.sqrt(sum_of_squared_weights / sum_of_weights_squared))
    return mean, std, mean_error

def weighted_mean_and_std(bin_centers, weights, squared_weights):
    mean = (bin_centers * weights).sum() / weights.sum()
    n = weights.sum()
    effective_n = (weights**2 / (squared_weights + (weights == 0))).sum()
    residuals = (((bin_centers - mean)**2 * weights)**2).sum()
    std = np.sqrt(1 / (effective_n - 1) * residuals) / (n / effective_n)
    sum_of_weights_squared = weights.sum()**2
    sum_of_squared_weights = squared_weights.sum()
    mean_error = std / (effective_n * np.sqrt(sum_of_squared_weights / sum_of_weights_squared))
    return mean, std, mean_error


def hist_mean_and_std(hist, axis=1):
    if hist.dimensions > 1:
        bin_centers = hist.binnings[axis].bin_centers
        weights = hist.values
        if hasattr(hist, "squared_values"):
            squared_weights = hist.squared_values
        else:
            squared_weights = weights
        return weighted_row_mean_and_std(bin_centers[1:-1], weights[1:-1,1:-1], squared_weights[1:-1,1:-1], axis=axis)
    elif hist.dimensions == 1:
        bin_centers = hist.binnings[0].bin_centers[1:-1]
        weights = hist.values[1:-1]
        if hasattr(hist, "squared_values"):
            squared_weights = hist.squared_values[1:-1]
        else:
            squared_weights = weights
        return weighted_mean_and_std(bin_centers, weights, squared_weights)
    raise NotImplementedError


def hist_percentile(hist, axis=1, percentile=0.95, bin_point="center", interpolate=False):
    def _get_bin_values(binning):
        if bin_point == "center":
            return binning.bin_centers
        elif bin_point == "low":
            return binning.edges[:-1]
        elif bin_point == "high":
            return binning.edges[1:]
        raise NotImplementedError
    if hist.dimensions == 1:
        assert axis == 0
        bin_values = _get_bin_values(hist.binnings[axis])
        cdf = np.cumsum(hist.values, axis=axis) / np.sum(hist.values)
        if interpolate:
            return np.interp(percentile, cdf, bin_values)
        else:
            return bin_values[np.argmin(np.abs(cdf - percentile), axis=axis)]
    if hist.dimensions == 2:
        bin_values = _get_bin_values(hist.binnings[axis])
        cdf = np.cumsum(hist.values, axis=axis) / np.expand_dims(np.sum(hist.values, axis=axis), axis=axis)
        if interpolate:
            cdf = np.moveaxis(cdf, 1 - axis, 0)
            return np.array([np.interp(percentile, c, bin_values) for c in cdf])
        else:
            return bin_values[np.argmin(np.abs(cdf - percentile), axis=axis)]
    raise NotImplementedError


def calculate_likelihood(distribution, values):
    norm = distribution.values.sum()
    return distribution.get(values) / norm


def random_powerlaw(E_min, E_max, gamma, n=100):
    ex = 1 - gamma
    return (np.random.random(n) * (E_max**ex - E_min**ex) + E_min**ex)**(1 / ex)

def integral_powerlaw(E_min, E_max, gamma, phi_0):
    ex = 1 - gamma
    return phi_0 / ex * (E_max**ex - E_min**ex)


def poisson_limit_lower(n, fraction):
    return scp_chi2.ppf(fraction, 2 * n) / 2

def poisson_limit_upper(n, fraction):
    return scp_chi2.ppf(fraction, 2 * (n + 1)) / 2

def poisson_interval(n, probability):
    remainder = 1 - probability
    lower_limit = poisson_limit_lower(n, remainder / 2)
    upper_limit = poisson_limit_lower(n, 1 - remainder / 2)
    return lower_limit, upper_limit

def poisson_limit(n, probability):
    return poisson_limit_upper(n, probability)

def n_sigmas_to_probability(n_sigmas):
    return 2 * scp_gaussian.cdf(n_sigmas) - 1


def approximate_upper_poisson_error(n, sigmas=1):
    return (n + 1) * (1 - 1 / (9 * (n + 1)) + sigmas / (3 * np.sqrt(n + 1)))**3 - n

poisson_parametrization_beta = {1: 0, 2: 0.062, 3: 0.222}
poisson_parametrization_gamma = {1: 0, 2: -2.19, 3: -1.88}

def approximate_lower_poisson_error(n, sigmas=1):
    beta = poisson_parametrization_beta[sigmas]
    gamma = poisson_parametrization_gamma[sigmas]
    return -n * (1 - 1 / (9 * np.maximum(n, 1e-7)) - sigmas / (3 * np.sqrt(np.maximum(n, 1e-7))) + beta * n**gamma)**3 + n
    

def smooth_additive(values, window=1):
    result = np.copy(values)
    count = np.ones_like(values)
    for shift in range(-window, window + 1):
        if shift < 0:
            result[:shift] += values[-shift:]
            count[:shift] += 1
        elif shift > 0:
            result[shift:] += values[:-shift]
            count[shift:] += 1
    return result / count


def calculate_correlation(x, y):
    x_central = x - np.mean(x)
    y_central = y - np.mean(y)
    return np.mean(x_central * y_central) / (np.std(x_central) * np.std(y_central))

def calculate_correlation_and_error(x, y):
    r = calculate_correlation(x, y)
    n = len(x)
    std = np.sqrt((1 - r**2) / (n - 2))
    return r, std


def draw_random_from_hist(histogram, size, seed=None, rng=None):
    from .utilities import transform_overflow_edges
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    probability = histogram.values / histogram.values.sum()
    cumulative = np.cumsum(probability)
    edges = transform_overflow_edges(histogram.binnings[0].edges)
    x_values = (edges[1:] + edges[:-1]) / 2
    first_index = np.max(np.arange(len(cumulative))[cumulative == 0]) if np.any(cumulative == 0) else 0
    last_index = np.min(np.arange(len(cumulative))[(1 - cumulative) < 1e-10])
    cumulative = cumulative[first_index:last_index + 1]
    x_values = x_values[first_index:last_index + 1]
    monotonic_sel = np.concatenate(([True], cumulative[1:] > cumulative[:-1]))
    cumulative = cumulative[monotonic_sel]
    x_values = x_values[monotonic_sel]
    spline = PchipInterpolator(cumulative, x_values)
    return spline(rng.random(size=size))


def randomize_hist(histogram):
    values = histogram.values
    errors = histogram.get_errors()
    print(values)
    print(errors)
    counts = values**2 / (errors**2 + (errors == 0))
    average_weights = errors**2 / (values + (values == 0))
    print(counts)
    print(average_weights)
    random_values = np.array([np.random.poisson(v) for v in counts])
    new_values = random_values * average_weights
    new_squared_values = random_values * average_weights**2
    print(new_values, new_squared_values)
    return WeightedHistogram(*histogram.binnings, values=new_values, squared_values=new_squared_values, labels=histogram.labels)


if __name__ == "__main__":
    print("1D")
    values = np.array([-1, 0, 1])
    counts = np.array([1, 2, 1])
    w1 = 1
    weights1 = counts * w1
    weights_squared1 = counts * w1**2
    w2 = 2
    weights2 = counts * w2
    weights_squared2 = counts * w2**2
    w3 = 0.5
    weights3 = counts * w3
    weights_squared3 = counts * w3**2

    print(weighted_mean_and_std(values, weights1, weights_squared1))
    print(weighted_mean_and_std(values, weights2, weights_squared2))
    print(weighted_mean_and_std(values, weights3, weights_squared3))

    print("2D")
    values = np.array([[-1, 0, 1], [1, 2, 3], [-1, 2, 5]])
    counts = np.array([[1, 2, 1], [1, 4, 1], [0, 3, 2]])
    print(row_mean_and_std(values, counts))
    w1 = 1
    weights1 = counts * w1
    weights_squared1 = counts * w1**2
    w2 = 2
    weights2 = counts * w2
    weights_squared2 = counts * w2**2
    w3 = 0.5
    weights3 = counts * w3
    weights_squared3 = counts * w3**2

    print(weighted_row_mean_and_std(values, weights1, weights_squared1))
    print(weighted_row_mean_and_std(values, weights2, weights_squared2))
    print(weighted_row_mean_and_std(values, weights3, weights_squared3))
