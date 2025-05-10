import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import Event_weight_prediction


def compute_event_weights_ctgRe(file, mask_file, SC_saved_path, wc_index, ctgRe_min=-1, ctgRe_max=1, num_points=9):
    """
    Computes event weights for a range of c_tGRe values.
    
    Parameters:
        file: HDF5 or ROOT file containing event data.
        SC_saved_path (str): Path to the stored SC values (NumPy file).
        ctgRe_min (float): Minimum value of c_tGRe.
        ctgRe_max (float): Maximum value of c_tGRe.
        num_points (int): Number of points to sample in the range.
        
    Returns:
        dict: A dictionary where keys are c_tGRe values and values are masked event weights.
    """
    # Load precomputed stored SC values
    SC_saved = np.load(SC_saved_path, allow_pickle=True)
    
    # Generate c_tGRe values
    ctgRe_values = np.linspace(ctgRe_min, ctgRe_max, num_points)
    
    # Dictionary to store event weights
    event_weights_ctgRe = {}
    
    for ctgRe_val in ctgRe_values:
        # Initialize a 16D Wilson coefficient vector
        wc_vector_ctgRe = [0] * 16  # Assume a 16-dimensional Wilson coefficient space
        wc_vector_ctgRe[wc_index] = ctgRe_val  # Modify only the relevant WC (c_tGRe)
        
        # Compute the event weight for the corresponding Wilson coefficient
        event_weight, _, _ = Event_weight_prediction.event_weights_lin_quad(SC_saved, wc_vector_ctgRe)
        
        # Store in dictionary
        event_weights_ctgRe[ctgRe_val] = event_weight
    
    # Reference SM weight
    weight_sm1 = file['mgWeights'][:, 200]
    
    
    # Apply the mask to all computed weights
    for ctgRe_val in event_weights_ctgRe:
        event_weights_ctgRe[ctgRe_val] = event_weights_ctgRe[ctgRe_val][mask_file]
    
    # Also mask the SM weight
    weight_sm1 = weight_sm1[mask_file]
    
    return event_weights_ctgRe, weight_sm1

#################################this is only for one observables#######################################################################
def compute_values_for_chi_square(x_obs, weight_sm, weight_ctgRe_dict, bins):
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

    eft_events = {}
    eft_errors = {}

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

        # Store results
        eft_events[ctgRe_value] = hist_eft
        eft_errors[ctgRe_value] = hist_eft_err

    return hist_sm, hist_sm_err, eft_events, eft_errors, bin_edges

def compute_chi_square(hist_sm, hist_sm_err, hist_eft, hist_eft_err, bins):
    """Compute the Chi-square statistic between SM and EFT histograms."""
    
    # Make sure both histograms have the same length
    assert len(hist_sm) == len(hist_eft), "Histograms should have the same length"
    
    # Compute Chi-square term for each bin
    chi_square = 0.0
    for i in range(len(hist_sm)):
        # Total uncertainty (sqrt of squares of individual errors)
        total_error = np.sqrt(hist_sm_err[i]**2 + hist_eft_err[i]**2)
        
        # Only consider bins where the uncertainty is non-zero to avoid division by zero
        if total_error > 0:
            chi_square += ((hist_sm[i] - hist_eft[i])**2) / total_error**2
    
    return chi_square/(bins-1) #return the chi squared divided by degree of freedom

