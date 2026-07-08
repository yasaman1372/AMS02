
import multiprocessing as mp

import numpy as np
from scipy.stats import norm, chi2, poisson, gaussian_kde
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, curve_fit

from .histograms import Histogram, WeightedHistogram
from .statistics import calculate_efficiency_and_error, draw_random_from_hist
from .utilities import make_kde_template

def loss_at_angle(loss, x, y, dx, dy):
    def _loss_at_r(r):
        return loss(x + r * dx, y + r * dy)
    return _loss_at_r

def loss_in_direction(loss, x0, dx):
    def _loss_at_r(r):
        return loss(x0 + r * dx)
    return _loss_at_r


def fit_linear(loss_func, target_value, precision=1e-6):
    min_r = 0
    max_r = 1
    while loss_func(max_r) < target_value:
        min_r = max_r
        max_r *= 2
    loss = loss_func((min_r + max_r) / 2)
    while np.isfinite(loss) and abs(loss - target_value) > precision:
        mid_r = (min_r + max_r) / 2
        if loss_func(mid_r) < target_value:
            min_r = mid_r
        else:
            max_r = mid_r
        loss = loss_func((min_r + max_r) / 2)
    if not np.isfinite(loss):
        print("Cannot calculate limit, loss not finite")
    best_r = (min_r + max_r) / 2
    best_fit = loss_func(best_r)
    diff = abs(best_fit - target_value)
    return best_r


def calculate_contour(loss_function, best_fit_parameters, loss_delta, sections=120):
    assert len(best_fit_parameters) == 2
    minimum_loss = loss_function(*best_fit_parameters)
    target_loss = minimum_loss + loss_delta
    param_x, param_y = best_fit_parameters

    contour_points = []
    for index in range(sections):
        angle = 2 * np.pi * index / sections
        dx_0 = np.cos(angle)
        dy_0 = np.sin(angle)
        effective_angle = np.arctan2(dx_0, dy_0 / 10)
        dx = np.cos(effective_angle)
        dy = np.sin(effective_angle)

        linear_loss = loss_at_angle(loss_function, param_x, param_y, dx, dy)
        target_r = fit_linear(linear_loss, target_loss)
        contour_points.append((param_x + dx * target_r, param_y + dy * target_r))
    contour_points.append(contour_points[0])
    return contour_points


def calculate_confidence_interval(confidence_level, loss, parameters, adjust_lower=False, compensation=0):
    loss_at_minimum = loss(*parameters)
    p = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    cl = chi2(1).ppf(p)
    l1 = loss_at_angle(loss, parameters[0], parameters[1], -1, compensation)
    r1 = fit_linear(l1, loss_at_minimum + cl)
    limit1 = parameters[0] - r1
    if adjust_lower:
        if limit1 < 0:
            delta_loss_at_zero = loss(0, parameters[1]) - loss_at_minimum
            p_less = chi2(1).cdf(delta_loss_at_zero) / 2
            cln = chi2(1).ppf(p - p_less)
            clo = cl
            cl = cln
            limit1 = 0
            l2 = loss_at_angle(loss, parameters[0], parameters[1], 1, -compensation)
    l2 = loss_at_angle(loss, parameters[0], parameters[1], 1, -compensation)
    r2 = fit_linear(l2, loss_at_minimum + cl)
    limit2 = parameters[0] + r2
    return limit1, limit2

def calculate_confidence_interval_1d(confidence_level, loss, parameters, parameter_index=0):
    loss_at_minimum = loss(*parameters)
    p = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    cl = chi2(1).ppf(p)
    param_values = list(parameters)
    def _loss(parameter_value):
        param_values[parameter_index] = parameter_value
        return loss(*param_values)
    l1 = loss_in_direction(_loss, parameters[parameter_index], 1)
    r1 = fit_linear(l1, loss_at_minimum + cl)
    limit1 = parameters[parameter_index] + r1
    l2 = loss_in_direction(_loss, parameters[parameter_index], -1)
    r2 = fit_linear(l2, loss_at_minimum + cl)
    limit2 = parameters[parameter_index] - r2
    return limit2, limit1


def make_pdf_and_ppf_from_hist(hist):
    assert hist.dimensions == 1
    binning = hist.binnings[0]
    x = binning.bin_centers[1:-1]
    edges = binning.edges[1:-1]
    pdf = PchipInterpolator(x, hist.values[1:-1] / hist.values[1:-1].sum())
    cdf = np.cumsum(hist.values / hist.values.sum())[:-1]
    if cdf[0] > 0:
        cdf[0] = 0
    if cdf[-1] < 1:
        cdf[-1] = 1
    monotonic = np.concatenate(([True], cdf[1:] - cdf[:-1] > 1e-100))
    finite = np.isfinite(cdf)
    ppf = PchipInterpolator(cdf[monotonic & finite], edges[monotonic & finite])
    return pdf, ppf

def calculate_confidence_limit_toy_mc_fit_linear(confidence_level, nsig_best, nbkg_best, signal_template_hist, background_hist, background_template_entries, kde_bandwidth, nsig_params, precision=0.01, test_significance=3, max_error=1e-3, verbose=True, nprocesses=1):

    background_values_x = background_hist.binnings[0].bin_centers[1:-1]
    background_values_y = background_hist.values[1:-1]
    background_errors_y = background_hist.get_errors()[1:-1]
    background_value_sel = background_values_y > 0
    background_values_x, background_values_y, background_errors_y = map(lambda a: a[background_value_sel], (background_values_x, background_values_y, background_errors_y))

    #def _make_segmented_linear(segments, min_x, max_x):
    #    def _segmented_linear(x, *args):
    #        slopes = args[:segments]
    #        cuts = np.concatenate(([-np.inf], np.cumsum(args[segments:2 * segments - 1])))
    #        offset = args[-1]
    #        offsets = [offset]
    #        for 
    def _segmented_linear_function(x, a1, a2, a3, c1, c2, b):
        c2 = c1 + c2
        b2 = (a1 - a2) * c1 + b
        b3 = (a2 - a1) * c2 + b2
        s1 = x < c1
        s2 = (x >= c1) & (x < c2)
        s3 = x >= c2
        y1 = a1 * x + b
        y2 = a2 * x + b2
        y3 = a3 * x + b3
        return np.maximum(y1 * s1 + y2 * s2 + y3 * s3, 0)
    bv_popt, bv_pcov = curve_fit(_segmented_linear_function, background_values_x, background_values_y, p0=(1, 0, 1, 0, 8, 20), sigma=background_errors_y, absolute_sigma=True)
    bv_perr = np.sqrt(np.diag(bv_pcov))
    bv_chisq = np.sum(((_segmented_linear_function(background_values_x, *bv_popt) - background_values_y) / background_errors_y)**2)
    bv_dof = len(background_values_x) - len(bv_popt)
    print(bv_popt, bv_perr, bv_chisq, bv_dof, bv_chisq / bv_dof)

    signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
    def _make_fit():
        background_mc_toy_events = draw_random_from_hist(background_hist, size=background_template_entries)
        kde = gaussian_kde(background_mc_toy_events, bw_method=kde_bandwidth)
        kde_template = make_kde_template(kde, signal_template_hist.binnings[0])
        kde_template_hist = Histogram(signal_template_hist.binnings[0], values=kde_template)
        background_pdf, background_ppf = make_pdf_and_ppf_from_hist(kde_template_hist)
        return ToyMCFit(signal_pdf=signal_pdf, background_pdf=background_pdf, signal_ppf=signal_ppf, background_ppf=background_ppf)
    return calculate_confidence_limit_toy_mc(confidence_level, nsig_best, nbkg_best, _make_fit, nsig_params=nsig_params, precision=precision, test_significance=test_significance, max_error=max_error, verbose=verbose, nprocesses=nprocesses)

def calculate_confidence_limit_toy_mc_redraw_background(confidence_level, nsig_best, nbkg_best, signal_template_hist, background_template_hist, background_hist, background_template_entries, kde_bandwidth, nsig_params, precision=0.01, test_significance=3, max_error=1e-3, verbose=True, nprocesses=1):
    signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
    #background_pdf, background_ppf = make_pdf_and_ppf_from_hist(background_template_hist)
    def _make_fit():
        ppf_background_mc_toy_events = draw_random_from_hist(background_hist, size=background_template_entries)
        ppf_kde = gaussian_kde(ppf_background_mc_toy_events, bw_method=kde_bandwidth)
        ppf_kde_template = make_kde_template(ppf_kde, signal_template_hist.binnings[0])
        ppf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=ppf_kde_template)
        _, new_background_ppf = make_pdf_and_ppf_from_hist(ppf_kde_template_hist)
        pdf_background_mc_toy_events = draw_random_from_hist(background_hist, size=background_template_entries)
        pdf_kde = gaussian_kde(pdf_background_mc_toy_events, bw_method=kde_bandwidth)
        pdf_kde_template = make_kde_template(pdf_kde, signal_template_hist.binnings[0])
        pdf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=pdf_kde_template)
        new_background_pdf, _ = make_pdf_and_ppf_from_hist(pdf_kde_template_hist)
        return ToyMCFit(signal_pdf=signal_pdf, background_pdf=new_background_pdf, signal_ppf=signal_ppf, background_ppf=new_background_ppf)
    return calculate_confidence_limit_toy_mc(confidence_level, nsig_best, nbkg_best, _make_fit, nsig_params=nsig_params, precision=precision, test_significance=test_significance, max_error=max_error, verbose=verbose, nprocesses=nprocesses)


def calculate_confidence_limit_toy_mc_supremum(confidence_level, nsig_best, nbkg_best, signal_template_hist, background_template_hist, background_hist, background_template_entries, kde_bandwidth, nsig_params, precision=0.01, test_significance=3, max_error=1e-3, verbose=True, nprocesses=1, samples=5, quantile=0.95):
    limits = []
    for _ in range(samples):
        signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
        background_pdf, background_ppf = make_pdf_and_ppf_from_hist(background_template_hist)
        ppf_background_mc_toy_events = draw_random_from_hist(background_hist, size=background_template_entries)
        ppf_kde = gaussian_kde(ppf_background_mc_toy_events, bw_method=kde_bandwidth)
        ppf_kde_template = make_kde_template(ppf_kde, signal_template_hist.binnings[0])
        ppf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=ppf_kde_template)
        _, new_background_ppf = make_pdf_and_ppf_from_hist(ppf_kde_template_hist)
        #pdf_background_mc_toy_events = draw_random_from_hist(background_hist, size=background_template_entries)
        #pdf_kde = gaussian_kde(pdf_background_mc_toy_events, bw_method=kde_bandwidth)
        #pdf_kde_template = make_kde_template(pdf_kde, signal_template_hist.binnings[0])
        #pdf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=pdf_kde_template)
        #new_background_pdf, _ = make_pdf_and_ppf_from_hist(pdf_kde_template_hist)
        def _make_fit():
            #return ToyMCFit(signal_pdf=signal_pdf, background_pdf=new_background_pdf, signal_ppf=signal_ppf, background_ppf=new_background_ppf)
            return ToyMCFit(signal_pdf=signal_pdf, background_pdf=background_pdf, signal_ppf=signal_ppf, background_ppf=new_background_ppf)
        limits.append(calculate_confidence_limit_toy_mc(confidence_level, nsig_best, nbkg_best, _make_fit, nsig_params=nsig_params, precision=precision, test_significance=test_significance, max_error=max_error, verbose=verbose, nprocesses=nprocesses))
        print(limits)
    print("limits", np.min(limits), np.mean(limits), np.median(limits), np.quantile(limits, quantile), np.max(limits))
    print(sorted(limits))
    return np.quantile(limits, 0.95)


def calculate_confidence_limit_toy_mc_from_hists(confidence_level, nsig_best, nbkg_best, signal_template_hist, background_template_hist, nsig_params, precision=0.01, test_significance=3, max_error=1e-3, verbose=True, nprocesses=1, record_steps=False):
    signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
    background_pdf, background_ppf = make_pdf_and_ppf_from_hist(background_template_hist)
    def _make_fit():
        return ToyMCFit(signal_pdf=signal_pdf, background_pdf=background_pdf, signal_ppf=signal_ppf, background_ppf=background_ppf)
    return calculate_confidence_limit_toy_mc(confidence_level, nsig_best, nbkg_best, _make_fit, nsig_params=nsig_params, precision=precision, test_significance=test_significance, max_error=max_error, verbose=verbose, nprocesses=nprocesses, record_steps=record_steps)


def calculate_combined_confidence_limit_toy_mc_redraw_background(confidence_level, rsig_best, nbkgs_best, nsig_parameters, signal_template_hists, background_hists, background_template_entries, kde_bandwidths, precision=0.01, test_significance=3, max_error=1e-3, ratio_factor=1, verbose=True, nprocesses=1):
    fits = []
    for signal_template_hist, background_hist, nbkg, kde_bandwidth in zip(signal_template_hists, background_hists, background_template_entries, kde_bandwidths):
        signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
        def _make_bkg_pdf_ppf():
            pdf_toy_events = draw_random_from_hist(background_hist, size=nbkg)
            pdf_kde = gaussian_kde(pdf_toy_events, bw_method=kde_bandwidth)
            pdf_kde_template = make_kde_template(pdf_kde, signal_template_hist.binnings[0])
            pdf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=pdf_kde_template)
            pdf_background_pdf, pdf_background_ppf = make_pdf_and_ppf_from_hist(pdf_kde_template_hist)
            ppf_toy_events = draw_random_from_hist(background_hist, size=nbkg)
            ppf_kde = gaussian_kde(ppf_toy_events, bw_method=kde_bandwidth)
            ppf_kde_template = make_kde_template(ppf_kde, signal_template_hist.binnings[0])
            ppf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=ppf_kde_template)
            ppf_background_pdf, ppf_background_ppf = make_pdf_and_ppf_from_hist(ppf_kde_template_hist)
            return pdf_background_pdf, ppf_background_ppf

        fits.append((signal_pdf, signal_ppf, _make_bkg_pdf_ppf))
    def _make_fits():
        _fits = []
        for signal_pdf, signal_ppf, make_bkg_pdf_ppf in fits:
            background_pdf, background_ppf = make_bkg_pdf_ppf()
            _fits.append(ToyMCFit(signal_pdf=signal_pdf, background_pdf=background_pdf, signal_ppf=signal_ppf, background_ppf=background_ppf))
        return _fits
    return calculate_combined_confidence_limit_toy_mc(confidence_level=confidence_level, rsig_best=rsig_best, nbkgs_best=nbkgs_best, fits_generator=_make_fits, nsig_params=nsig_parameters, precision=precision, test_significance=test_significance, max_error=max_error, ratio_factor=ratio_factor, verbose=verbose, nprocesses=nprocesses)


def calculate_combined_confidence_limit_toy_mc_from_hists(confidence_level, rsig_best, nbkgs_best, nsig_parameters, signal_template_hists, background_template_hists, precision=0.01, test_significance=3, max_error=1e-3, ratio_factor=1, verbose=True, nprocesses=1, record_steps=False):
    fits = []
    for signal_template_hist, background_template_hist in zip(signal_template_hists, background_template_hists):
        signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
        background_pdf, background_ppf = make_pdf_and_ppf_from_hist(background_template_hist)
        fits.append(ToyMCFit(signal_pdf=signal_pdf, background_pdf=background_pdf, signal_ppf=signal_ppf, background_ppf=background_ppf))
    def _make_fits():
        return fits
    return calculate_combined_confidence_limit_toy_mc(confidence_level=confidence_level, rsig_best=rsig_best, nbkgs_best=nbkgs_best, fits_generator=_make_fits, nsig_params=nsig_parameters, precision=precision, test_significance=test_significance, max_error=max_error, ratio_factor=ratio_factor, verbose=verbose, nprocesses=nprocesses, record_steps=record_steps)


def calculate_combined_confidence_limit_toy_mc_supremum(confidence_level, rsig_best, nbkgs_best, nsig_parameters, signal_template_hists, background_template_hists, background_hists, background_template_entries, kde_bandwidths, precision=0.01, test_significance=3, max_error=1e-3, ratio_factor=1, verbose=True, nprocesses=1, samples=5, quantile=0.95):
    limits = []
    for _ in range(samples):
        fits = []
        for signal_template_hist, background_template_hist, background_hist, nbkg, kde_bandwidth in zip(signal_template_hists, background_template_hists, background_hists, background_template_entries, kde_bandwidths):
            signal_pdf, signal_ppf = make_pdf_and_ppf_from_hist(signal_template_hist)
            background_pdf, background_ppf = make_pdf_and_ppf_from_hist(background_template_hist)
            #pdf_toy_events = draw_random_from_hist(background_hist, size=nbkg)
            #pdf_kde = gaussian_kde(pdf_toy_events, bw_method=kde_bandwidth)
            #pdf_kde_template = make_kde_template(pdf_kde, signal_template_hist.binnings[0])
            #pdf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=pdf_kde_template)
            #new_background_pdf, _ = make_pdf_and_ppf_from_hist(pdf_kde_template_hist)
            ppf_toy_events = draw_random_from_hist(background_hist, size=nbkg)
            ppf_kde = gaussian_kde(ppf_toy_events, bw_method=kde_bandwidth)
            ppf_kde_template = make_kde_template(ppf_kde, signal_template_hist.binnings[0])
            ppf_kde_template_hist = Histogram(signal_template_hist.binnings[0], values=ppf_kde_template)
            _, new_background_ppf = make_pdf_and_ppf_from_hist(ppf_kde_template_hist)
            fits.append(ToyMCFit(signal_pdf=signal_pdf, background_pdf=background_pdf, signal_ppf=signal_ppf, background_ppf=new_background_ppf))
        def _make_fits():
            return fits
        limits.append(calculate_combined_confidence_limit_toy_mc(confidence_level=confidence_level, rsig_best=rsig_best, nbkgs_best=nbkgs_best, fits_generator=_make_fits, nsig_params=nsig_parameters, precision=precision, test_significance=test_significance, max_error=max_error, ratio_factor=ratio_factor, verbose=verbose, nprocesses=nprocesses))
        print(limits)
    print("limits", np.min(limits), np.mean(limits), np.median(limits), np.quantile(limits, quantile), np.max(limits))
    print(sorted(limits))
    return np.quantile(limits, quantile)




class ToyMCFit:
    def __init__(self, signal_pdf, background_pdf, signal_ppf, background_ppf):
        self.signal_pdf = signal_pdf
        self.background_pdf = background_pdf
        self.signal_ppf = signal_ppf
        self.background_ppf = background_ppf

    def create_sample(self, nsig, nbkg, rng):
        return np.concatenate((self.signal_ppf(rng.random(nsig)), self.background_ppf(rng.random(nbkg))))

    def likelihood(self, nsig, nbkg, sample):
        p = nsig * self.signal_pdf(sample) + nbkg * self.background_pdf(sample)
        return 2 * (nsig + nbkg - np.sum(np.log(p + 1e-200)))

    def fit(self, sample):
        nsig_guess = 1
        nbkg_guess = len(sample) - 1
        def _loss(args):
            nsig, nbkg = args
            return self.likelihood(nsig, nbkg, sample)
        opt = minimize(_loss, (nsig_guess, nbkg_guess), bounds=((0, None), (0, None)))
        nsig_fit, nbkg_fit = opt.x
        return nsig_fit, nbkg_fit

def fit_combined(fits_and_samples, ratio_factor):
    guess = [1e-8 * ratio_factor] + [len(sample) - 1 for _, sample, _ in fits_and_samples]
    bounds = [(0, None)] + [(0, None) for _ in fits_and_samples]
    def _loss(args):
        rsig, *nbkgs = args
        return sum(fit.likelihood(rsig_to_nsig_nocorr(rsig, nsig_params, ratio_factor=ratio_factor), nbkg, sample) for (fit, sample, nsig_params), nbkg in zip(fits_and_samples, nbkgs))
    opt = minimize(_loss, guess, bounds=bounds)
    rsig_fit, *nbkgs_fit = opt.x
    return rsig_fit


def rsig_to_nsig(rsig, parameters, ratio_factor):
    slope, offset, npos = parameters
    nsig_raw = rsig * npos / ratio_factor
    nsig = np.maximum(slope * nsig_raw + offset, 0)
    sel = nsig_raw >= 0
    return sel * nsig + np.invert(sel) * nsig_raw

def rsig_to_nsig_nocorr(rsig, parameters, ratio_factor):
    slope, offset, npos = parameters
    nsig_raw = rsig * npos / ratio_factor
    return nsig_raw

def nsig_to_nsig(nsig_raw, parameters):
    slope, offset = parameters
    nsig = np.maximum(slope * nsig_raw + offset, 0)
    sel = nsig_raw >= 0
    return sel * nsig + np.invert(sel) * nsig_raw


def calculate_fraction_part(args):
    n, rsig_best, fits, rsig, nbkgs, nsig_parameters, ratio_factor = args
    above = 0
    above_zero = 0
    rng = np.random.default_rng()
    for _ in range(n):
        samples = [fit.create_sample(rng.poisson(rsig_to_nsig(rsig, nsig_params, ratio_factor=ratio_factor)), rng.poisson(nbkg), rng) for fit, nbkg, nsig_params in zip(fits, nbkgs, nsig_parameters)]
        fits_and_samples = list(zip(fits, samples, nsig_parameters))
        rsig_fit = fit_combined(fits_and_samples, ratio_factor=ratio_factor)
        above_zero += (rsig_fit > 0)
        above += (rsig_fit > rsig_best)
    return above, n

def calculate_combined_confidence_limit_toy_mc(confidence_level, rsig_best, nbkgs_best, fits_generator, nsig_params, precision=0.01, test_significance=3, max_error=1e-3, ratio_factor=1, verbose=True, nprocesses=1, min_samples=500, record_steps=False):
    if confidence_level >= 1:
        target_fraction = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    else:
        target_fraction = confidence_level
    rsig_min = 0
    rsig_max = 1e-7 * ratio_factor

    steps = []

    def _calculate_fraction(rsig, nbkgs, fits, nsig_params):
        total = 0
        above = 0
        def _gen():
            for _ in range(nprocesses):
                yield (min_samples, rsig_best * ratio_factor, fits, rsig, nbkgs, nsig_params, ratio_factor)
        with mp.Pool(nprocesses) as pool:
            while True:
                for k, n in pool.imap_unordered(calculate_fraction_part, _gen()):
                    above += k
                    total += n
                fraction, fraction_error = calculate_efficiency_and_error(above, total)
                residual = (fraction - target_fraction) / fraction_error
                if verbose:
                    print(f"{rsig/ratio_factor:>10.4g}: {above:>6} / {total:>6} = {fraction:>6.4f}±{fraction_error:>6.4f} (res={residual:.1f})", end="\r")
                if abs(residual) >= test_significance or (fraction_error < max_error and abs(fraction - target_fraction) < max_error):
                    if verbose:
                        print(f"{rsig/ratio_factor:>5.4g} -> f={fraction:.4f} ± {fraction_error:.4f} (res={residual:.1f})" + (20 * " "))
                    if record_steps:
                        steps.append((rsig / ratio_factor, nbkgs, fraction, fraction_error, residual))
                    return fraction, fraction_error, residual
    fits = fits_generator()
    min_fraction, min_fraction_error, min_significance = _calculate_fraction(rsig_min, nbkgs_best, fits, nsig_params)
    max_fraction, max_fraction_error, max_significance = _calculate_fraction(rsig_max, nbkgs_best, fits, nsig_params)
    while max_significance <= test_significance:
        rsig_max *= 10
        max_fraction, max_fraction_error, max_significance = _calculate_fraction(rsig_max, nbkgs_best, fits, nsig_params)
    #print(min_significance, max_significance)
    assert min_significance < -test_significance and max_significance > test_significance
    result = None
    while True:
        fits = fits_generator()
        pivot = (rsig_min + rsig_max) / 2
        if (rsig_max - rsig_min) / (rsig_min + rsig_max) <= precision:
            result = pivot / ratio_factor
            break
        fraction, fraction_error, residual = _calculate_fraction(pivot, nbkgs_best, fits, nsig_params)
        if abs(residual) >= test_significance:
            if residual > 0:
                rsig_max = pivot
            else:
                rsig_min = pivot
        else:
            result = pivot / ratio_factor
            break
    if record_steps:
        return result, steps
    return result

def calculate_individual_fraction_part(args):
    n, nsig_best, fit, nsig, nbkg, nsig_params = args
    above = 0
    rng = np.random.default_rng()
    for _ in range(n):
        sample = fit.create_sample(rng.poisson(nsig_to_nsig(nsig, nsig_params)), rng.poisson(nbkg), rng)
        nsig_fit, nbkg_fit = fit.fit(sample)
        above += (nsig_fit > nsig_best)
    return above, n

def calculate_confidence_limit_toy_mc(confidence_level, nsig_best, nbkg_best, fit_generator, nsig_params, precision=0.01, test_significance=3, max_error=1e-3, verbose=False, nprocesses=1, record_steps=False):
    if confidence_level >= 1:
        target_fraction = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    else:
        target_fraction = confidence_level
    nsig_min = max(0, nsig_best - 2 * np.sqrt(nsig_best))
    nsig_max = min(max(max(nbkg_best / 2, nsig_best + 2 * np.sqrt(nsig_best)), 10), 100)

    steps = []
    
    def _calculate_fraction(nsig, nbkg, fit, nsig_params):
        total = 0
        above = 0
        def _gen():
            for _ in range(nprocesses):
                yield (100, nsig_best, fit, nsig, nbkg, nsig_params)
        with mp.Pool(nprocesses) as pool:
            while True:
                for k, n in pool.imap_unordered(calculate_individual_fraction_part, _gen()):
                    above += k
                    total += n
                fraction, fraction_error = calculate_efficiency_and_error(above, total)
                residual = (fraction - target_fraction) / fraction_error
                if verbose:
                    print(f"{nsig:>10.4g}: {above:>6} / {total:>6} = {fraction:>6.4f}±{fraction_error:>6.4f} (res={residual:.1f})", end="\r")
                if abs(residual) >= test_significance or (fraction_error < max_error and abs(fraction - target_fraction) < max_error):
                    if verbose:
                        print(f"{nsig:>5.4g} -> f={fraction:.4f} ± {fraction_error:.4f} (res={residual:.1f})" + (20 * " "))
                    if record_steps:
                        steps.append((nsig, nbkg, fraction, fraction_error, residual))
                    return fraction, fraction_error, residual
    fit = fit_generator()
    min_fraction, min_fraction_error, min_significance = _calculate_fraction(nsig_min, nbkg_best, fit, nsig_params)
    max_fraction, max_fraction_error, max_significance = _calculate_fraction(nsig_max, nbkg_best, fit, nsig_params)
    #print(min_significance, max_significance)
    while max_significance < test_significance:
        nsig_max *= 10
        max_fraction, max_fraction_error, max_significance = _calculate_fraction(nsig_max, nbkg_best, fit, nsig_params)
    assert min_significance < -test_significance and max_significance > test_significance
    pivot_ratio = 0.5
    result = None
    while True:
        fit = fit_generator()
        if nsig_max - nsig_min <= precision:
            result = (nsig_min + nsig_max) / 2
            break
        pivot = nsig_min + (nsig_max - nsig_min) * pivot_ratio
        fraction, fraction_error, residual = _calculate_fraction(pivot, nbkg_best, fit, nsig_params)
        if abs(residual) >= test_significance:
            if residual > 0:
                nsig_max = pivot
            else:
                nsig_min = pivot
            pivot_ratio = 0.5
        else:
            if fraction_error <= max_error and abs(fraction - target_fraction) < fraction_error and abs(residual) < 1:
                result = pivot
                break
            if residual > 0:
                pivot_ratio = 1 - (1 - pivot_ratio) * 0.5
            else:
                pivot_ratio = pivot_ratio * 0.5
    if record_steps:
        return result, steps
    return result


def calculate_toy_mc_limit_cut_supremum(confidence_level, events_above_cut, mc_events_above_cut, mc_average_weight, mc_average_weight_error, precision=0.01, test_significance=3, max_error=1e-3, verbose=False, ntries=1000, quantile=0.95):
    limits = []
    rng = np.random.default_rng()
    for _ in range(ntries):
        n_bkg = rng.poisson(mc_events_above_cut) * np.maximum(rng.normal(mc_average_weight, mc_average_weight_error), 0)
        limits.append(calculate_toy_mc_limit_cut(confidence_level=confidence_level, events_above_cut=events_above_cut, mc_events_above_cut=n_bkg, mc_average_weight=1, mc_average_weight_error=0, randomize_bkg=False, precision=precision, test_significance=test_significance, max_error=max_error, verbose=verbose))
    
    #print(sorted(limits))
    return np.quantile(limits, quantile)


def calculate_toy_mc_limit_cut(confidence_level, events_above_cut, mc_events_above_cut, mc_average_weight, mc_average_weight_error, precision=0.01, test_significance=3, max_error=1e-3, randomize_bkg=True, verbose=False):
    if confidence_level >= 1:
        target_fraction = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    else:
        target_fraction = confidence_level
    rng = np.random.default_rng()
    def _draw(nsig, size=10000):
        if randomize_bkg:
            n_bkg = rng.poisson(mc_events_above_cut, size) * np.maximum(rng.normal(mc_average_weight, mc_average_weight_error, size), 0)
        else:
            n_bkg = mc_events_above_cut
        n_on = rng.poisson(nsig + n_bkg, size)
        return np.sum(n_on > events_above_cut), size
    def _calculate_fraction(nsig):
        above = 0
        total = 0
        while True:
            k, n = _draw(nsig)
            above += k
            total += n
            fraction, fraction_error = calculate_efficiency_and_error(above, total)
            residual = (fraction - target_fraction) / fraction_error
            if verbose:
                print(f"{nsig:>10.4g}: {above:>6} / {total:>6} = {fraction:>6.4f}±{fraction_error:>6.4f} (res={residual:.1f})", end="\r")
            if abs(residual) >= test_significance or (fraction_error < max_error and abs(fraction - target_fraction) < max_error):
                if verbose:
                    print(f"{nsig:>5.4g} -> f={fraction:.4f} ± {fraction_error:.4f} (res={residual:.1f})" + (20 * " "))
                return fraction, fraction_error, residual
    nsig_min = 0
    nsig_max = 10
    min_fraction, min_fraction_error, min_significance = _calculate_fraction(nsig_min)
    max_fraction, max_fraction_error, max_significance = _calculate_fraction(nsig_max)
    while max_significance < test_significance:
        if verbose:
            print("nsig_max", nsig_max, max_fraction)
        nsig_max *= 10
        max_fraction, max_fraction_error, max_significance = _calculate_fraction(nsig_max)
    if verbose:
        print(nsig_min, nsig_max)
    pivot_ratio = 0.5
    while True:
        if nsig_max - nsig_min <= precision:
            return (nsig_min + nsig_max) / 2
        pivot = nsig_min + (nsig_max - nsig_min) * pivot_ratio
        fraction, fraction_error, residual = _calculate_fraction(pivot)
        if abs(residual) >= test_significance:
            if residual > 0:
                nsig_max = pivot
            else:
                nsig_min = pivot
            pivot_ratio = 0.5
        else:
            if fraction_error <= max_error and abs(fraction - target_fraction) < fraction_error and abs(residual) < 1:
                return pivot
            if residual > 0:
                pivot_ratio = 1 - (1 - pivot_ratio) * 0.5
            else:
                pivot_ratio = pivot_ratio * 0.5


def calculate_combined_confidence_limit_toy_mc_cut(confidence_level, samples, precision=0.01, test_significance=3, max_error=1e-3, randomize_bkg=True, verbose=False):
    if confidence_level >= 1:
        target_fraction = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    else:
        target_fraction = confidence_level
    rng = np.random.default_rng()
    def _draw(rsig, size=10000):
        above = 0
        total = 0
        for negative_events, positive_events, mc_events, mc_to_iss_factor, mc_to_iss_factor_error in samples:
            n_sig = rsig * positive_events
            if randomize_bkg:
                n_bkg = rng.poisson(mc_events, size) * np.maximum(rng.normal(mc_to_iss_factor, mc_to_iss_factor_error, size), 0)
            else:
                n_bkg = mc_events
            n_on = rng.poisson(n_sig + n_bkg, size)
            above += np.sum(n_on > negative_events)
            total += size
        return above, total
    def _calculate_fraction(rsig):
        above = 0
        total = 0
        while True:
            k, n = _draw(rsig)
            above += k
            total += n
            fraction, fraction_error = calculate_efficiency_and_error(above, total)
            residual = (fraction - target_fraction) / fraction_error
            if verbose:
                print(f"{rsig:>10.4g}: {above:>6} / {total:>6} = {fraction:>6.4f}±{fraction_error:>6.4f} (res={residual:.1f})", end="\r")
            if abs(residual) >= test_significance or (fraction_error < max_error and abs(fraction - target_fraction) < max_error):
                if verbose:
                    print(f"{rsig:>5.4g} -> f={fraction:.4f} ± {fraction_error:.4f} (res={residual:.1f})" + (20 * " "))
                return fraction, fraction_error, residual
    rsig_min = 0
    rsig_max = 1e-6
    min_fraction, min_fraction_error, min_significance = _calculate_fraction(rsig_min)
    max_fraction, max_fraction_error, max_significance = _calculate_fraction(rsig_max)
    while max_significance < test_significance:
        rsig_max *= 10
        max_fraction, max_fraction_error, max_significance = _calculate_fraction(rsig_max)
    if verbose:
        print(rsig_min, rsig_max)
    pivot_ratio = 0.5
    while True:
        if (rsig_max - rsig_min) / (rsig_max + rsig_min) <= precision:
            return (rsig_min + rsig_max) / 2
        pivot = rsig_min + (rsig_max - rsig_min) * pivot_ratio
        fraction, fraction_error, residual = _calculate_fraction(pivot)
        if abs(residual) >= test_significance:
            if residual > 0:
                rsig_max = pivot
            else:
                rsig_min = pivot
            pivot_ratio = 0.5
        else:
            if fraction_error <= max_error and abs(fraction - target_fraction) < fraction_error and abs(residual) < 1:
                return pivot
            if residual > 0:
                pivot_ratio = 1 - (1 - pivot_ratio) * 0.5
            else:
                pivot_ratio = pivot_ratio * 0.5



def test_coverage(cl, n_sig_min, n_sig_max, n_bkg, mu_sig, mu_bkg, sigma_sig, sigma_bkg, n_test=1000):
    def _make_gaussian_fits():
        signal = norm(mu_sig, sigma_sig)
        background = norm(mu_bkg, sigma_bkg)
        return ToyMCFit(signal_pdf=signal.pdf, background_pdf=background.pdf, signal_ppf=signal.ppf, background_ppf=background.ppf)
    rng = np.random.default_rng()
    nsigs = rng.uniform(low=n_sig_min, high=n_sig_max, size=n_test)
    limits = []
    above = []
    nsig_ms = []
    for nsig in nsigs:
        nsig_m = rng.poisson(nsig)
        print("n", nsig, nsig_m)
        limit = calculate_confidence_limit_toy_mc(confidence_level=cl, nsig_best=nsig_m, nbkg_best=n_bkg, fit_generator=_make_gaussian_fits, nsig_params=(1, 0), precision=0.01, test_significance=5, max_error=0.0025, verbose=False, nprocesses=mp.cpu_count())
        limits.append(limit)
        above.append(limit >= nsig)
        nsig_ms.append(nsig_m)
        print("n", nsig, "l", limit)
        print(f"f = {np.sum(above)}/{len(above)} = {np.sum(above)/len(above):.4f}")
        
    coverage = np.sum(above) / len(above)
    print(f"CL: {cl:.4f}, coverage: {coverage:.4f}")
    return coverage, nsigs, np.array(limits), np.array(nsig_ms)


if __name__ == "__main__":
    #print(calculate_toy_mc_limit_cut(confidence_level=0.95, events_above_cut=2, mc_events_above_cut=6, mc_average_weight=0.3, verbose=True))
    #print(calculate_toy_mc_limit_cut(confidence_level=0.95, events_above_cut=0, mc_events_above_cut=0, mc_average_weight=1, verbose=True))
    #print(calculate_toy_mc_limit_cut(confidence_level=0.95, events_above_cut=4, mc_events_above_cut=2, mc_average_weight=2, verbose=True))

    test_coverage(cl=0.95, n_sig_min=0, n_sig_max=5, n_bkg=100, mu_sig=5, mu_bkg=0, sigma_sig=0.5, sigma_bkg=1.5, n_test=1000)

    #print("Zero zero far")
    #test_cl(cl=0.95, n_sig=0, n_bkg=0, mu_sig=10, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)
    #print("Zero ten far")
    #test_cl(cl=0.95, n_sig=0, n_bkg=10, mu_sig=10, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)
    #print("Zero 100 far")
    #test_cl(cl=0.95, n_sig=0, n_bkg=100, mu_sig=10, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)
    #print("Zero 100 medium")
    #test_cl(cl=0.95, n_sig=0, n_bkg=100, mu_sig=5, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)
    #print("Zero 100 close")
    #test_cl(cl=0.95, n_sig=0, n_bkg=100, mu_sig=2, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)
    #print("Zero 100 very close")
    #test_cl(cl=0.95, n_sig=0, n_bkg=100, mu_sig=1.5, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)
    #print("10 100 medium")
    #test_cl(cl=0.95, n_sig=10, n_bkg=100, mu_sig=5, mu_bkg=0, sigma_sig=0.5, sigma_bkg=2)

