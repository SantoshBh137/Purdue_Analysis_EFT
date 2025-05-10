import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_delta_chi2(test_statistics,wc_name, ylabel, bins, observable):
    """
    Plots the limit plot based on delta chi-square statistics.
    
    Parameters:
    test_statistics (dict): Dictionary with c_tGRe values as keys and chi-square values as values.
    bins (int): Bin information.
    observable (str): Name of the observable.
    """
    # Convert to arrays for plotting
    ctgRe_values = np.array(list(test_statistics.keys()))
    test_stat_values = np.array(list(test_statistics.values()))
    
    # Interpolating the Δχ² function for smoothness
    interp_func = interp1d(ctgRe_values, test_stat_values, kind='cubic', fill_value="extrapolate")
    
    # Fine grid for smooth curve
    fine_ctgRe = np.linspace(min(ctgRe_values), max(ctgRe_values), 300)
    fine_delta_chi2 = interp_func(fine_ctgRe)

    # Compute Δχ² = χ² - χ²_min
    best_fit_stat = min(fine_delta_chi2)
    delta_chi2_values = fine_delta_chi2 - best_fit_stat  # Compute Δχ²
    
    # Find best-fit value (minimum of Δχ²)
    best_fit_idx = np.argmin(delta_chi2_values)
    best_fit_ctgRe = fine_ctgRe[best_fit_idx]
    
    # Define 1σ and 2σ confidence intervals
    sigma1_mask = delta_chi2_values <= 1
    sigma2_mask = (delta_chi2_values > 1) & (delta_chi2_values <= 4)
    
    sigma1_range = fine_ctgRe[sigma1_mask]
    sigma2_range = fine_ctgRe[sigma2_mask]
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(fine_ctgRe, delta_chi2_values, 'k-', label=r'$\Delta\chi^2$')
    
    # Shading for 1σ and 2σ intervals
    plt.fill_between(sigma2_range, 0, 10, color='#FFCB00', label=r'$\pm2\sigma$')  # Orange
    plt.fill_between(sigma1_range, 0, 10, color='#00CB00', label=r'$\pm1\sigma$')  # Green
    
    up_sig1 = max(sigma1_range) - best_fit_ctgRe
    down_sig1 = best_fit_ctgRe - min(sigma1_range)
    
    up_sig2 = max(sigma2_range) - best_fit_ctgRe
    down_sig2 = best_fit_ctgRe - min(sigma2_range)
    
    best_fit_str = rf"$c_{{tGRe}} = {best_fit_ctgRe:.3f}^{{+{up_sig1:.3f}}}_{{-{down_sig1:.3f}}}$"
    
    # Dashed line for best fit
    plt.axvline(best_fit_ctgRe, color='k', linestyle='--', label=f"{best_fit_str}")
    
    # Labels
    plt.xlabel(rf"{wc_name}", fontsize=16)
    plt.ylabel(rf"{ylabel}", fontsize=16)
    plt.title(rf"({observable}), bin={bins}")
    plt.ylim(0, 7)  # Set a lower bound at 0 since Δχ² is non-negative
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()
    
    return best_fit_ctgRe, up_sig1, down_sig1, up_sig2,down_sig2

