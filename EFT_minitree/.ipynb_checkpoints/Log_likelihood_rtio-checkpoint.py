import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import Event_weight_prediction



def compute_test_statistics_1D(x_obs, weight_sm, weight_ctgRe_dict, bins):
    """Compute test statistics for different ctGRe values."""

    # Convert Awkward arrays to NumPy
    x_obs_np = ak.to_numpy(x_obs)
    weight_sm_np = ak.to_numpy(weight_sm)

    # Determine bin edges based on observable range
    min_val, max_val = np.min(x_obs_np), np.max(x_obs_np)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_widths = np.diff(bin_edges)  # Width of each bin

    # Compute SM histogram (acts as observed data)
    hist_sm, _ = np.histogram(x_obs_np, bins=bin_edges, weights=weight_sm_np, density=True)

    # Compute raw event counts per bin (unweighted)
    counts_sm, _ = np.histogram(x_obs_np, bins=bin_edges)
    total_weighted_events_sm = np.sum(weight_sm_np)

    # Corrected statistical uncertainty (normalized histogram)
    hist_sm_err = np.sqrt(counts_sm) / (total_weighted_events_sm * bin_widths)

    test_statistics = {}

    # Iterate over different ctGRe values
    for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
        weight_eft_np = ak.to_numpy(weight_eft)

        # Compute EFT histogram
        hist_eft, _ = np.histogram(x_obs_np, bins=bin_edges, weights=weight_eft_np, density=True)

        # Compute raw event counts per bin (unweighted for EFT)
        counts_eft, _ = np.histogram(x_obs_np, bins=bin_edges)
        total_weighted_events_eft = np.sum(weight_eft_np)

        # Corrected statistical uncertainty for EFT histogram
        hist_eft_err = np.sqrt(counts_eft) / (total_weighted_events_eft * bin_widths)

        total_error= np.sqrt(hist_eft_err**2+hist_sm_err**2)
        test_statistic=-2*np.sum((hist_sm - hist_eft) ** 2 / (2 * total_error ** 2)+ np.log(hist_eft / hist_sm))
        test_statistics[ctgRe_value] = test_statistic / (bins - 1)
    return test_statistics