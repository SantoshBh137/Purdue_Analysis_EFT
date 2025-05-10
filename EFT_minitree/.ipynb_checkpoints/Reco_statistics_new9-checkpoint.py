import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import Event_weight_prediction


def compute_covariance_matrix(data, bin_edges, n_bootstrap):
    """Estimate the covariance matrix using bootstrap resampling."""
    n_bins = len(bin_edges) - 1
    bootstrap_samples = np.zeros((n_bootstrap, n_bins))

    data_np = ak.to_numpy(data)

    for i in range(n_bootstrap):
        resampled_data = np.random.choice(data_np, size=len(data_np), replace=True)
        hist, _ = np.histogram(resampled_data, bins=bin_edges, density=True)
        bootstrap_samples[i] = hist

    covariance_matrix = np.cov(bootstrap_samples, rowvar=False)
    return covariance_matrix

def compute_chi_square(hist_obs, hist_eft, covariance_matrix):
    """Compute the reduced chi-square statistic using the covariance matrix."""
    residuals = hist_obs - hist_eft
    try:
        inv_cov_matrix = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.linalg.pinv(covariance_matrix)  # Use pseudo-inverse if singular

    chi_square = residuals.T @ inv_cov_matrix @ residuals

    # Degrees of freedom: number of bins minus the number of estimated parameters
    dof = len(hist_obs) - 1  # Assuming one parameter is estimated

    # Calculate reduced chi-square
    reduced_chi_square = chi_square / dof if dof > 0 else np.nan

    return reduced_chi_square



def analyze_data(observed_data, eft_data, weight_ctgRe_dict, eft_reco_weights, bins, n_bootstrap):

    chi_square_results = {}

    for obs in observed_data.keys():
        # Convert observed and EFT data to numpy arrays
        x_obs_np = ak.to_numpy(observed_data[obs])
        x_eft_np = ak.to_numpy(eft_data[obs])

        # Define bin edges based on observed data range
        min_val, max_val = np.min(x_obs_np), np.max(x_obs_np)
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Compute observed histogram (normalized)
        hist_obs, _ = np.histogram(x_obs_np, bins=bin_edges, density=True)

        # Compute covariance matrix for the observed data
        covariance_matrix = compute_covariance_matrix(x_obs_np, bin_edges, n_bootstrap)

        for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
            # Convert EFT weights to numpy array
            weight_eft_np = ak.to_numpy(weight_eft)

            # Compute EFT histogram (weighted and normalized)
            hist_eft, _ = np.histogram(
                x_eft_np,
                bins=bin_edges,
                weights=weight_eft_np * ak.to_numpy(eft_reco_weights),
                density=True
            )

            # Compute chi-square statistic for the current ctgRe value
            chi_square = compute_chi_square(hist_obs, hist_eft, covariance_matrix)

            # Store the result as a single numerical value
            if ctgRe_value not in chi_square_results:
                chi_square_results[ctgRe_value] = {}
            chi_square_results[ctgRe_value] = chi_square

    return chi_square_results
