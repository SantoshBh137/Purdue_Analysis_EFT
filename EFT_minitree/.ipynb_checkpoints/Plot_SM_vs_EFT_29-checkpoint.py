import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def plot_sm_vs_eft_histograms(observables, bin_edges, hist_sm_list, hist_sm_err_list, eft_events, eft_errors, x_label, wc_name, legend_sizes):
    """
    Plots SM scatter plot with error bars and EFT step histograms for different Wilson coefficient values.
    Uses Data (hist_sm) as the reference for the ratio plot.
    """
    # Extract histograms correctly for each observable
    eft_histograms = {observable: {} for observable in observables}
    eft_histogram_errors = {observable: {} for observable in observables}

    for ctgRe_value, hist_list in eft_events.items():
        for i, observable in enumerate(observables):
            if i < len(hist_list):  # Ensure index is within bounds
                eft_histograms[observable][ctgRe_value] = hist_list[i]
                eft_histogram_errors[observable][ctgRe_value] = eft_errors[ctgRe_value][i]

    # Set up CMS style
    plt.style.use(hep.style.CMS)

    # Create subplots: Main plot + Ratio plot
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(legend_sizes, legend_sizes - 4), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Sort available Wilson coefficient values
    all_ctGRe_values = sorted(eft_histograms[observables[0]].keys())

    # Select exactly 5 EFT values that actually exist in the dictionary
    num_selected = min(len(all_ctGRe_values), 5)
    selected_ctGRe_values = np.linspace(min(all_ctGRe_values), max(all_ctGRe_values), num_selected)

    # Find closest actual available EFT values
    selected_ctGRe_values = [min(all_ctGRe_values, key=lambda x: abs(x - v)) for v in selected_ctGRe_values]

    # Set specific colors: Black (Data), Red, Green, Blue, Orange (EFT)
    colors = ['blue', 'green','red','magenta', 'orange']  # EFT colors
    eft_color_map = {ctGRe: colors[i % len(colors)] for i, ctGRe in enumerate(selected_ctGRe_values)}

    for observable in observables:
        # Get the SM histogram and error bars for this observable
        hist_sm = hist_sm_list[observables.index(observable)]
        hist_sm_err = hist_sm_err_list[observables.index(observable)]

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

        # Plot SM scatter plot with connecting line (Black)
        ax.errorbar(bin_centers, hist_sm, yerr=hist_sm_err, fmt='o', color='black', capsize=4, label="Data")

        # Plot EFT histograms for selected Wilson coefficient values
        for ctGRe in selected_ctGRe_values:
            eft_hist = eft_histograms[observable][ctGRe]
            eft_hist_err = eft_histogram_errors[observable][ctGRe]

            # Use specific colors for EFT curves
            color = eft_color_map[ctGRe]
            ax.hist(bin_centers, bins=bin_edges, weights=eft_hist, histtype="step", linestyle="--", color=color, label=f"EFT ({wc_name}={ctGRe:.2f})")

            # Compute Ratio: EFT / Data
            ratio = np.divide(eft_hist, hist_sm, out=np.ones_like(hist_sm), where=hist_sm != 0)
            ratio_err = np.divide(eft_hist_err, hist_sm, out=np.zeros_like(hist_sm), where=hist_sm != 0)

            # Plot ratio with uncertainty
            ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='o', color=color,linestyle='-', capsize=4, label=f"EFT ({wc_name}={ctGRe:.2f})")

        # Labels and settings
        ax.set_ylabel("Normalized Events")
        ax.legend(fontsize=legend_sizes)
        ax.grid()
        ax.set_ylim(0, None)

        ax_ratio.set_xlabel(x_label)
        ax_ratio.set_ylabel("EFT / Data")
        ax_ratio.axhline(1, color='gray', linestyle='--', linewidth=1)  # Reference line
        ax_ratio.set_ylim(0.8, 1.2)
        ax_ratio.grid()
        #ax_ratio.legend(fontsize=legend_sizes, loc="best")

    plt.tight_layout()
    plt.show()
