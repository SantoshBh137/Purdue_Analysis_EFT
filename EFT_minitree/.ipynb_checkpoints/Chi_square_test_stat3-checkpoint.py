import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import Event_weight_prediction

def compute_values_for_chi_square(x_obs_list, weight_sm, weight_ctgRe_dict, bins):
    """Compute test statistics for different ctGRe values over multiple observables."""
    
    hist_sm_list = []  # List to store SM histograms for each observable
    hist_sm_err_list = []  # List to store SM errors for each observable
    eft_events = {}
    eft_errors = {}

    # Iterate over each observable in the list
    for x_obs in x_obs_list:
        # Convert Awkward arrays to NumPy
        x_obs_np = ak.to_numpy(x_obs_list[x_obs])
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

        hist_sm_list.append(hist_sm)
        hist_sm_err_list.append(hist_sm_err)

        # Iterate over different ctGRe values for EFT samples
        for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
            weight_eft_np = ak.to_numpy(weight_eft)

            # Compute EFT histogram
            hist_eft, _ = np.histogram(x_obs_np, bins=bin_edges, weights=weight_eft_np, density=True)

            # Compute raw event counts per bin (unweighted for EFT)
            counts_eft, _ = np.histogram(x_obs_np, bins=bin_edges)
            total_weighted_events_eft = np.sum(weight_eft_np)

            # Corrected statistical uncertainty for EFT histogram
            hist_eft_err = np.sqrt(counts_eft) / (total_weighted_events_eft * bin_widths)

            # Store results
            if ctgRe_value not in eft_events:
                eft_events[ctgRe_value] = []
                eft_errors[ctgRe_value] = []
            eft_events[ctgRe_value].append(hist_eft)
            eft_errors[ctgRe_value].append(hist_eft_err)

    return hist_sm_list, hist_sm_err_list, eft_events, eft_errors, bin_edges

def compute_chi_square(hist_sm_list, hist_sm_err_list, eft_events, eft_errors, bins):
    """Compute the Chi-square statistic between SM and EFT histograms for multiple observables."""
    
    # Initialize the total chi-square value
    total_chi_square = 0.0
    n_observables = len(hist_sm_list)  # Number of observables

    # Iterate over each observable
    for hist_sm, hist_sm_err, hist_eft, hist_eft_err in zip(hist_sm_list, hist_sm_err_list, eft_events, eft_errors):
        # Ensure histograms have the same length
        assert len(hist_sm) == len(hist_eft), "Histograms should have the same length"
        
        # Initialize chi-square for this observable
        chi_square = 0.0

        # Iterate over bins
        for i in range(len(hist_sm)):
            # Total uncertainty (sqrt of squares of individual errors)
            #total_error = np.sqrt(hist_sm_err[i]**2 + hist_eft_err[i]**2)
            
            # Only consider bins where the uncertainty is non-zero to avoid division by zero
            if total_error > 0:
                chi_square += ((hist_sm[i] - hist_eft[i])**2) / hist_eft[i]

        # Add chi-square for this observable to total
        total_chi_square += chi_square / (bins - 1)  # Normalize by degrees of freedom

    # Return the final chi-square value
    return total_chi_square / n_observables  # Average over observables
