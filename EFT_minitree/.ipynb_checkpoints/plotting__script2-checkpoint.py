import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak

def plot_weight_validation(weight_data_dict, bins_space=(0.1, 10), n_bins=60):
    """
    Function to plot multiple histograms for different weight datasets with arbitrary spacing and logarithmic scaling.

    Parameters:
    - weight_data_dict: A dictionary where keys are labels for datasets and values are arrays of weight data.
    - bins_space: Tuple defining the min and max values for the weight range (default: (0.1, 10)).
    - n_bins: The number of bins to use for the histograms (default: 60).
    """
    # Define the bin edges for the histograms in log scale
    bins = np.logspace(np.log10(bins_space[0]), np.log10(bins_space[1]), n_bins)

    # Load CMS style
    plt.style.use(hep.style.CMS)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histograms for each weight dataset
    for label, weights in weight_data_dict.items():
        # Calculate the histogram values
        hist, edges = np.histogram(weights, bins=bins, density=False)

        # Find the bin centers (midpoints)
        bin_centers = 0.5 * (edges[1:] + edges[:-1])

        # Check if it's the dataset to plot as scatter
        if label == "ctGRe2_calculated":  # Replace "scatter_dataset" with the desired label
            ax.scatter(bin_centers, hist, label=label, marker='x')
        else:
            ax.hist(weights, bins=bins, histtype='step', label=label, density=False)

    # Add labels, title, and legend
    ax.set_xlabel('Weight')
    ax.set_ylabel('Number of Events')
    ax.set_xscale('log')  # Set x-axis to log scale
    ax.legend()

    # Add CMS Preliminary header
    hep.cms.text("Preliminary", ax=ax)

    # Show the plot
    plt.show()


def plot_observables_ReweightedSM_EFT(obs, xlabel, title, weights_dict, num_bins):
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Determine bin edges based on observable range
    min_val, max_val = ak.min(obs), ak.max(obs)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Color cycle for different weights
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Extract reweighted SM as the reference distribution
    reweighted_sm_weights = weights_dict.get('SM', None)
    if reweighted_sm_weights is None:
        raise ValueError("Key 'SM' must be in weights_dict")

    # Plot histogram for reweighted SM
    counts_reweighted, _, _ = ax.hist(
        obs, bins=bin_edges, label="SM", color=colors[0], 
        weights=reweighted_sm_weights, histtype='step', density=True
    )

    # Initialize dictionary for storing ratios
    ratio_dict = {}

    # Loop through all EFT weights (excluding reweighted SM)
    for i, (label, weight_values) in enumerate(weights_dict.items()):
        if label == 'SM':  # Skip SM itself
            continue
        
        # Assign colors dynamically
        color = colors[(i + 1) % len(colors)]

        # Plot histogram for each EFT weight
        counts_eft, _, _ = ax.hist(
            obs, bins=bin_edges, label=label, color=color, 
            weights=weight_values, histtype='step', density=True
        )

        # Compute ratio EFT / Reweighted SM
        ratio_dict[label] = np.divide(
            counts_eft, counts_reweighted, 
            out=np.ones_like(counts_eft), where=counts_reweighted != 0
        )

        # Plot ratio
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_ratio.plot(bin_centers, ratio_dict[label], color=color, marker='o', linestyle='-')


    # Adjust ratio plot limits
    ax_ratio.set_ylim(0.6, 1.4)

    # Labels and CMS style
    ax.set_ylabel("Normalized Events")
    ax.set_title(title)
    ax.legend(loc='best')
    plt.style.use(hep.style.CMS)

    ax_ratio.set_xlabel(xlabel)
    ax_ratio.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax_ratio.set_ylabel("Ratio")

    plt.tight_layout()
    plt.show()

def plot_observables_ReweightedSM_SM(obs, obs_sm_file, xlabel, title, weights_dict, num_bins):
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Determine common bin edges based on min and max values across both datasets
    min_val = min(ak.min(obs), ak.min(obs_sm_file))
    max_val = max(ak.max(obs), ak.max(obs_sm_file))
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Color cycle for different weights
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Extract weights
    weight_sm = weights_dict['reweighted_SM']

    # Plot histogram for reweighted SM
    counts_reweighted, _, _ = ax.hist(obs, bins=bin_edges, label="Reweighted SM", color=colors[0], 
                                      weights=weight_sm, histtype='step', density=True)

    # Data (SM file)
    counts_sm_file, _, _ = ax.hist(obs_sm_file, bins=bin_edges, label="Data", color="black", 
                                   histtype='step', linestyle='dashed', density=True)

    # Compute ratio (avoid division by zero)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ratio_sm = np.divide(counts_sm_file, counts_reweighted, out=np.ones_like(counts_sm_file), 
                         where=counts_reweighted != 0)

    # Plot ratio without uncertainty bars
    ax_ratio.plot(bin_centers, ratio_sm, 'o', color='black', label="SM (Jason) / Reweighted SM")

    # Adjust ratio plot limits
    ax_ratio.set_ylim(0.8, 1.2)

    # Labels and CMS style
    ax.set_ylabel("Normalized Events")
    ax.set_title(title)
    ax.legend(loc='best')
    plt.style.use(hep.style.CMS)

    ax_ratio.set_xlabel(xlabel)
    ax_ratio.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax_ratio.set_ylabel("Ratio")

    plt.tight_layout()
    plt.show()
    
    return counts_reweighted, counts_sm_file