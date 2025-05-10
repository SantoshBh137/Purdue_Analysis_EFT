import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import scipy.stats

def compute_values_for_poisson_PLR(observed_data, eft_data, weight_ctgRe_dict, eft_reco_weights, bins):
    """Compute histograms and counts for Poisson-based profile likelihood ratio calculation."""
    
    hist_obs_list = []  
    counts_obs_list = []  
    eft_events = {}

    for obs in observed_data.keys():
        x_obs_np = ak.to_numpy(observed_data[obs])
        x_eft_np = ak.to_numpy(eft_data[obs])

        min_val, max_val = np.min(x_obs_np), np.max(x_obs_np)
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Compute observed histogram (unweighted)
        counts_obs, _ = np.histogram(x_obs_np, bins=bin_edges,density=True)
        hist_obs_list.append(counts_obs)
        counts_obs_list.append(counts_obs)

        # Compute EFT histograms for different weights
        for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
            weight_eft_np = ak.to_numpy(weight_eft)

            # Compute EFT expected counts
            counts_eft, _ = np.histogram(x_eft_np, bins=bin_edges, weights=weight_eft_np * ak.to_numpy(eft_reco_weights),density=True)

            if ctgRe_value not in eft_events:
                eft_events[ctgRe_value] = []
            eft_events[ctgRe_value].append(counts_eft)

    return hist_obs_list, eft_events, bin_edges



def compute_poisson_PLR( hist_obs_list, eft_events, bins):
    """Compute the Poisson Profile Likelihood Ratio (PLR) statistic for each EFT hypothesis."""
    
    n_observables = len(hist_obs_list)

    # Convert counts_obs_list to a NumPy array if it's a list
    counts_obs_array = np.array(hist_obs_list, dtype=float)

    # Find the best-fit hypothesis (the one with the highest likelihood)
    best_fit_counts = None
    best_fit_ctgRe = None
    max_likelihood = -np.inf

    for ctgRe_value, counts_eft_list in eft_events.items():
        # Convert counts_eft_list to NumPy array
        counts_eft_array = np.array(counts_eft_list, dtype=float)

        # Compute Poisson log-likelihood
        log_likelihood = np.sum(
            counts_obs_array * np.log(counts_eft_array + 1e-15) - counts_eft_array
        )

        if log_likelihood > max_likelihood:
            max_likelihood = log_likelihood
            best_fit_counts = counts_eft_array
            best_fit_ctgRe = ctgRe_value

    # Dictionary to store PLR values for each EFT hypothesis
    plr_values = {}

    # Compute the Poisson Profile Likelihood Ratio for each EFT hypothesis
    for ctgRe_value, counts_eft_list in eft_events.items():
        # Convert counts_eft_list to NumPy array
        counts_eft_array = np.array(counts_eft_list, dtype=float)

        # Avoid division by zero
        best_fit_counts = np.where(best_fit_counts == 0, 1e-15, best_fit_counts)
        counts_eft_array = np.where(counts_eft_array == 0, 1e-15, counts_eft_array)

        # Compute the PLR for this EFT hypothesis
        plr = -2*np.sum(
            counts_obs_array * np.log(counts_eft_array / best_fit_counts) - 
            (counts_eft_array - best_fit_counts)
        )

        # Normalize and store the PLR value
        plr_values[ctgRe_value] = plr

    return plr_values, best_fit_ctgRe  # Return dictionary of PLR values for each EFT hypothesis
