import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import os

mplhep.style.use("CMS")

observable_labels = {
    "b1k": r"$\cos\theta_1^k$", "b2k": r"$\cos\theta_2^k$",
    "b1r": r"$\cos\theta_1^r$", "b2r": r"$\cos\theta_2^r$",
    "b1n": r"$\cos\theta_1^n$", "b2n": r"$\cos\theta_2^n$",
    "c_kk": r"$\cos\theta_1^k \cos\theta_2^k$", "c_kr": r"$\cos\theta_1^k \cos\theta_2^r$", "c_kn": r"$\cos\theta_1^k \cos\theta_2^n$",
    "c_rk": r"$\cos\theta_1^r \cos\theta_2^k$", "c_rr": r"$\cos\theta_1^r \cos\theta_2^r$", "c_rn": r"$\cos\theta_1^r \cos\theta_2^n$",
    "c_nk": r"$\cos\theta_1^n \cos\theta_2^k$", "c_nr": r"$\cos\theta_1^n \cos\theta_2^r$", "c_nn": r"$\cos\theta_1^n \cos\theta_2^n$",
    "ll_cHel": r"$\cos\phi$",
    "llbar_delta_phi": r"$\Delta\phi_{ll}$",
    "top_pt": r"$p_T^\mathrm{top}$", "top_phi": r"$\phi^\mathrm{top}$",
    "top_rapidity": r"$y^\mathrm{top}$", "top_eta": r"$\eta^\mathrm{top}$",
    "ttbar_pt": r"$p_T^{t\bar{t}}$", "ttbar_phi": r"$\phi^{t\bar{t}}$",
    "ttbar_rapidity": r"$y^{t\bar{t}}$", "ttbar_eta": r"$\eta^{t\bar{t}}$",
    "ttbar_delta_phi": r"$\Delta\phi_{t\bar{t}}$", "ttbar_delta_eta": r"$\Delta\eta_{t\bar{t}}$",
    "gen_ttbar_mass": r"$m_{t\bar{t}}$"
}

class ObservablePlotter:
    def __init__(self, file_dict, plot_dir="plots"):
        self.file_dict = file_dict
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

    def plot(self, observable, step=0, include_uncertainty=False, save_name=None, binning_dict=None):
        assert step in [0, 8], "step must be 0 (GEN) or 8 (RECO)"
        tree_name = f"ttBar_treeVariables_step{step}"
        obs_prefix = "gen_" if step == 0 else ""
        obs_name = obs_prefix + observable

        obs_dict = {}
        weights = {}

        for label, file in self.file_dict.items():
            with uproot.open(file) as f:
                tree = f[tree_name]
                obs = tree[obs_name].array()
                obs_np = ak.to_numpy(obs)
        
                if "final_weight" in tree.keys():
                    weight = ak.to_numpy(tree["final_weight"].array())
                    print(f"[INFO] '{label}' uses 'final_weight'")
        
                elif "finalWeight" in tree.keys():
                    final_w = ak.to_numpy(tree["finalWeight"].array())
                    if step == 0:
                        base_w = ak.to_numpy(tree["trueLevelWeight"].array())
                        weight = final_w * base_w
                        print(f"[INFO] '{label}' uses 'finalWeight × trueLevelWeight'")
                    else:
                        base_w = ak.to_numpy(tree["eventWeight"].array())
                        weight = final_w * base_w
                        print(f"[INFO] '{label}' uses 'finalWeight × eventWeight'")
                else:
                    raise ValueError(f"No valid weight branch found in file '{file}'")
        
                obs_dict[label] = obs_np
                weights[label] = weight


        fig, (ax, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(10, 8))
        colors = ['black', 'red', 'blue', 'green']

        reference_label = list(obs_dict.keys())[0]
        reference_counts = None
        reference_errors = None
        bin_edges = None
        global_max = 0

        bins = binning_dict.get(observable, 60) if binning_dict else 60
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

        if "ttbar_mass" in observable:
            ax.set_yscale("log")
            ax.set_xlim(300, 1500)
            ax_ratio.set_xlim(300, 1500)
            ax_ratio.set_ylim(0.8, 2)
        else:
            ax.set_ylim(0, 1.1 * global_max)
            ax_ratio.set_ylim(0.5, 1.5)

        ax.legend(loc='best')
        label_text = "GEN" if step == 0 else "RECO"
        mplhep.cms.label(f"Work in progress ({label_text})", data=True, ax=ax, loc=0)

        xlabel = observable_labels.get(observable, observable)
        ax_ratio.set_xlabel(xlabel)
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.axhline(1, color='gray', linestyle='--')
        ax_ratio.legend().remove()

        plt.tight_layout()
        if save_name:
            plt.savefig(f"{self.plot_dir}/{save_name}.png")
        plt.show()
