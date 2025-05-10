import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import Event_weight_prediction



def compute_test_statistics_2D(x_obs, y_obs, weight_sm, weight_ctgRe_dict, bins):
    """Compute test statistics using the likelihood ratio method for different ctGRe values"""
 
    # Convert Awkward arrays to NumPy
    x_obs_np = ak.to_numpy(x_obs)
    y_obs_np = ak.to_numpy(y_obs)
    weight_sm_np = ak.to_numpy(weight_sm)
        
    #xmin_val, xmin_val = ak.min(x_obs_np), ak.max(x_obs_np)
    x_bin_edges = np.linspace(350, 1000, bins + 1)
    
    #ymin_val, ymin_val = ak.min(y_obs_np), ak.max(y_obs_np)
    y_bin_edges = np.linspace(-1, 1, bins + 1)
    
    # Compute SM histogram (acts as observed data)
    hist_sm, _, _ = np.histogram2d(x_obs_np, y_obs_np, bins=[x_bin_edges, y_bin_edges], weights=weight_sm_np, density=True)
    test_statistics = {}

    # Iterate over different ctGRe values
    for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
            
        weight_eft_np = ak.to_numpy(weight_eft)

        # Compute EFT histogram for this ctgRe (expected under EFT)
        hist_eft, _, _ = np.histogram2d(x_obs_np, y_obs_np, bins=[x_bin_edges, y_bin_edges], weights=weight_eft_np, density=True)
        
        # Avoid division by zero and log(0)
        epsilon = 1e-15
        hist_sm = np.where(hist_sm == 0, epsilon, hist_sm)
        hist_eft = np.where(hist_eft == 0, epsilon, hist_eft)

        # Compute test statistic: Wilks' theorem based likelihood ratio
        test_statistic = -2 * np.sum(hist_sm * np.log(hist_eft / hist_sm) + (hist_sm - hist_eft))

        # Store test statistic
        test_statistics[ctgRe_value] = test_statistic

    return test_statistics

def compute_test_statistics_1D(obs, weight_sm, weight_ctgRe_dict, bins):
    """Compute test statistics using the likelihood ratio method for different ctGRe values in 1D"""
    
    # Convert Awkward arrays to NumPy
    obs_np = ak.to_numpy(obs)
    weight_sm_np = ak.to_numpy(weight_sm)
    
    # Define bin edges
    obs_min, obs_max = np.min(obs_np), np.max(obs_np)
    bin_edges = np.linspace(obs_min, obs_max, bins + 1)
    
    # Compute SM histogram (acts as observed data)
    hist_sm, _ = np.histogram(obs_np, bins=bin_edges, weights=weight_sm_np, density=True)
    
    test_statistics = {}

    # Iterate over different c_tGRe values
    for ctgRe_value, weight_eft in weight_ctgRe_dict.items():
        
        weight_eft_np = ak.to_numpy(weight_eft)

        # Compute EFT histogram for this ctGRe (expected under EFT)
        hist_eft, _ = np.histogram(obs_np, bins=bin_edges, weights=weight_eft_np, density=True)

        # Avoid division by zero and log(0)
        epsilon = 1e-15
        hist_sm = np.where(hist_sm == 0, epsilon, hist_sm)
        hist_eft = np.where(hist_eft == 0, epsilon, hist_eft)

        # Compute test statistic: Wilks' theorem based likelihood ratio
        test_statistic = -2 * np.sum(hist_sm * np.log(hist_eft / hist_sm) + (hist_sm - hist_eft))

        # Store test statistic
        test_statistics[ctgRe_value] = test_statistic

    return test_statistics