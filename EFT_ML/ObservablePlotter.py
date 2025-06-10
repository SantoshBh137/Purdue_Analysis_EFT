import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import os

mplhep.style.use("CMS")


class ObservablePlotter:
    def __init__(self, file_dict, plot_dir="plots"):
        self.file_dict = file_dict
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

    def plot(self, observable, step=0, include_uncertainty=False, save_name=None, bins=None):

        assert step in [0, 8], "step must be 0 (GEN) or 8 (RECO)"

        tree_name = f"ttBar_treeVariables_step{step}"
        obs_prefix = "gen_" if step == 0 else ""
        obs_name = observable if step != 0 else ("gen_" + observable if not observable.startswith("gen_") else observable)

        obs_dict = {}
        weights = {}
        for label, file in self.file_dict.items():
            with uproot.open(file) as f:
                tree = f[tree_name]
                obs = tree[obs_name].array()
                wgt = tree["final_weight"].array()
                obs_dict[label] = ak.to_numpy(obs)
                weights[label] = ak.to_numpy(wgt)

        fig, (ax, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(10, 8))
        colors = ['black', 'red', 'blue', 'green']

        reference_label = list(obs_dict.keys())[0]
        reference_counts = None
        reference_errors = None
        bin_edges = None
        global_max = 0

        
        for i, label in enumerate(obs_dict):
            obs = obs_dict[label]
            weight = weights[label]
    
            bins_used = np.histogram_bin_edges(obs, bins=bins)
            counts, _ = np.histogram(obs, bins=bins_used, weights=weight, density=True)
            sumw, _ = np.histogram(obs, bins=bins_used, weights=weight)
            sumw2, _ = np.histogram(obs, bins=bins_used, weights=np.square(weight))
    
            bin_widths = np.diff(bins_used)
            errors = np.sqrt(sumw2) / (sumw * bin_widths)
            errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
            bin_centers = 0.5 * (bins_used[:-1] + bins_used[1:])
            global_max = max(global_max, np.max(counts + errors))
    
            ax.step(bins_used, np.append(counts, counts[-1]), where='post', color=colors[i % len(colors)], linewidth=2, label=label)
            ax.vlines([bins_used[0], bins_used[-1]], 0, [counts[0], counts[-1]], color=colors[i % len(colors)], linewidth=2)
            if include_uncertainty:
                ax.errorbar(bin_centers, counts, yerr=errors, fmt='o', color=colors[i % len(colors)], markersize=4, capsize=2)
    
            if i == 0:
                reference_counts = counts
                reference_errors = errors
                bin_edges = bins_used
            else:
                ratio = np.divide(counts, reference_counts, out=np.zeros_like(counts), where=reference_counts != 0)
                rel_err_ref = np.divide(reference_errors, reference_counts, out=np.zeros_like(reference_errors), where=reference_counts != 0)
                rel_err = np.divide(errors, counts, out=np.zeros_like(errors), where=counts != 0)
                ratio_err = ratio * np.sqrt(rel_err**2 + rel_err_ref**2)
                ax_ratio.step(bins_used, np.append(ratio, ratio[-1]), where='post', color=colors[i % len(colors)])
                if include_uncertainty:
                    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='o', color=colors[i % len(colors)], markersize=4, capsize=2)
    
        for edge in [bin_edges[0], bin_edges[-1]]:
            ax.axvline(edge, color='gray', linestyle=':', linewidth=1)
            ax_ratio.axvline(edge, color='gray', linestyle=':', linewidth=1)
    
        ax.set_ylabel("Normalized Events")

        if "ttbar_mass" in obs_name.lower():
            ax.set_yscale("log")
            ax.set_xlim(300, 1500)
            ax_ratio.set_xlim(300, 1500)
            ax_ratio.set_ylim(0.8, 2)
        else:
            ax.set_ylim(0, 1.1 * global_max)
            ax_ratio.set_ylim(0.75, 1.25)


        ax.legend(loc='best')
        mplhep.cms.label("Work in progress", data=True, ax=ax, loc=0)

        # Set label depending on step
        if step == 0:
            xlabel = "gen_" + observable.replace("gen_", "")
        else:
            xlabel = "reco_" + observable.replace("gen_", "")
        
        ax_ratio.set_xlabel(xlabel)

        ax_ratio.set_ylabel("Ratio")
        ax_ratio.axhline(1, color='gray', linestyle='--')
        #ax_ratio.set_ylim(0.8, 1.2)
        ax_ratio.legend().remove()

        plt.tight_layout()
        if save_name:
            plt.savefig(f"{self.plot_dir}/{save_name}.png")
        plt.show()
