import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import Event_weight_prediction


def compute_values_for_chi_square(observed_data, eft_data, weight_ctgRe_dict, eft_reco_weights, bins):
    """Compute test statistics using separate observed and EFT datasets with proper statistical errors."""
    
    hist_obs_list = []  
    hist_obs_err_list = []  
    eft_events = {}
    eft_errors = {}

    for obs in observed_data.keys():
        x_obs_np = ak.to_numpy(observed_data[obs])
        x_eft_np = ak.to_numpy(eft_data[obs])

        min_val, max_val = np.min(x_obs_np), np.max(x_obs_np)
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        bin_widths = np.diff(bin_edges)

        # Compute observed histogram (unweighted)
        hist_obs, _ = np.histogram(x_obs_np, bins=bin_edges, density=True)
        counts_obs, _ = np.histogram(x_obs_np, bins=bin_edges)
        
        # Properly normalize observed histogram error
        hist_obs_err = np.sqrt(counts_obs) / (counts_obs.sum() * bin_widths)

        hist_obs_list.append(hist_obs)
        hist_obs_err_list.append(hist_obs_err)

        # Compute EFT histograms for different weights
        for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
            weight_eft_np = ak.to_numpy(weight_eft)

            hist_eft, _ = np.histogram(x_eft_np, bins=bin_edges, weights=weight_eft_np* ak.to_numpy(eft_reco_weights), density=True)
            counts_eft, _ = np.histogram(x_eft_np, bins=bin_edges)
            
            # Compute statistical uncertainty for weighted EFT events
            weight_squared, _ = np.histogram(x_eft_np, bins=bin_edges, weights=weight_eft_np* ak.to_numpy(eft_reco_weights))
            total_weighted_events_eft = np.sum(weight_squared)
            
            hist_eft_err = np.sqrt(weight_squared) / (total_weighted_events_eft * bin_widths)

            if ctgRe_value not in eft_events:
                eft_events[ctgRe_value] = []
                eft_errors[ctgRe_value] = []
            eft_events[ctgRe_value].append(hist_eft)
            eft_errors[ctgRe_value].append(hist_eft_err)

    return hist_obs_list, hist_obs_err_list, eft_events, eft_errors, bin_edges

def compute_chi_square(hist_sm_list, hist_sm_err_list, eft_events, eft_errors):
    """
    Compute the Chi-square statistic between Standard Model (SM) and Effective Field Theory (EFT) histograms,
    considering statistical uncertainties, and return the total errors for each bin.

    Parameters:
    - hist_sm_list: List of arrays containing SM histogram values for each observable.
    - hist_sm_err_list: List of arrays containing SM histogram statistical uncertainties for each observable.
    - eft_events: List of arrays containing EFT histogram values for each observable.
    - eft_errors: List of arrays containing EFT histogram statistical uncertainties for each observable.

    Returns:
    - total_chi_square: The computed chi-square statistic.
    - total_errors: List of arrays containing total errors for each bin and observable.
    """
    total_chi_square = 0.0
    total_bins = 0  # Track total number of bins for correct DoF normalization
    total_errors = []  # To store total errors for each bin and observable

    for hist_sm, hist_sm_err, hist_eft, hist_eft_err in zip(hist_sm_list, hist_sm_err_list, eft_events, eft_errors):
        chi_square = 0.0
        num_bins = len(hist_sm)  # Number of bins in this observable
        errors = np.zeros(num_bins)  # To store total errors for current observable

        for i in range(num_bins):
            # Total statistical uncertainty for the current bin
            total_error = np.sqrt(hist_sm_err[i]**2 + hist_eft_err[i]**2)
            errors[i] = total_error  # Store the total error

            # Avoid division by zero issues
            if total_error > 0:
                chi_square += ((hist_sm[i] - hist_eft[i])**2) / total_error**2

        total_chi_square += chi_square
        total_bins += num_bins
        total_errors.append(errors)  # Store errors for the current observable

    # Normalize by total degrees of freedom (bins - 1 per observable)
    dof = total_bins - len(hist_sm_list)
    if dof > 0:
        total_chi_square /= dof

    return total_chi_square, total_errors