import os, uproot
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import os
import sys
sys.path.append('/depot/cms/top/bhanda25/Purdue_Analysis_EFT/Purdue_Analysis_EFT/EFT_minitree')
import Event_weight_prediction1

class EFTReweighter:
    def __init__(self, directory_path, eras, channels, mass_regions, cross_sections, struct_const_dir, step):
        self.directory_path = directory_path
        self.cross_sections = cross_sections
        self.struct_const_dir = struct_const_dir
        self.step = step

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.file_paths = [
            os.path.join(directory_path,
                         f"spinCorrInput_{era}/Nominal/{channel}/{channel}_ttto2l2nu_jet_smeft_mtt_{region}_{era}.root")
            for era in eras
            for channel in channels
            for region in mass_regions
        ]

        struct_step = "gen" if step == 0 else "reco"
        self.struct_paths = [
            os.path.join(struct_const_dir,
                         f"saved_sc_{era}/Nominal/{channel}_ttto2l2nu_jet_smeft_mtt_{region}_{era}_struct_{struct_step}.npy")
            for era in eras
            for channel in channels
            for region in mass_regions
        ]

        self.weights = self._compute_file_weights()
        self.final_observables = {}
        self.structure_constants = None

    def _compute_file_weights(self):
        weights = {}
        grouped = {k: [] for k in self.cross_sections.keys()}
        for path in self.file_paths:
            for region in grouped:
                if region in path:
                    grouped[region].append(path)

        for region, files in grouped.items():
            total_events = 0
            for f in files:
                with uproot.open(f) as file:
                    total_events += file[f"ttBar_treeVariables_step{self.step}"].num_entries
            for f in files:
                weights[f] = self.cross_sections[region] / total_events if total_events > 0 else 0

        max_weight = max(weights.values()) if weights else 1
        weights = {k: v / max_weight for k, v in weights.items()}

        return weights

    def load_structure_constants(self):
        paths = self.struct_paths
        total_rows = 0
        example_shape = None
        valid_paths = []

        for path in paths:
            if os.path.exists(path):
                shape = np.load(path, mmap_mode="r").shape
                total_rows += shape[0]
                example_shape = shape
                valid_paths.append(path)
            else:
                print(f"Missing: {path}")

        if example_shape is None:
            raise RuntimeError("No valid structure constant files found.")

        final_shape = (total_rows, example_shape[1])
        struct_array = np.empty(final_shape, dtype=np.float32)

        current_index = 0
        for path in tqdm(valid_paths, desc=f"Loading step{self.step} structure constants"):
            data = np.load(path)
            n = data.shape[0]
            struct_array[current_index:current_index + n] = data
            current_index += n

        self.structure_constants = torch.tensor(struct_array, dtype=torch.float32, device=self.device)
        print("Structure constants loaded.")

    def load_observables(self):
        collected = []

        for file_path in tqdm(self.file_paths, desc="Loading observables", unit="file"):
            with uproot.open(file_path) as file:
                tree = file[f"ttBar_treeVariables_step{self.step}"]
                keys = [k for k in tree.keys() if (self.step == 0 and k.startswith("gen_")) or (self.step == 8 and not k.startswith("gen_"))]
                extra = "trueLevelWeight" if self.step == 0 else "eventWeight"
                arrays = tree.arrays(keys + [extra])
                collected.append({k: arrays[k] for k in arrays.fields})

        self.final_observables = {
            k: ak.concatenate([obs[k] for obs in collected]) for k in collected[0].keys()
        }
        self.collected = collected

    def _base_weights(self):
        observables_list = self.collected
        example_key = next(iter(observables_list[0]))
        return ak.concatenate([
            np.full(len(d[example_key]), self.weights[self.file_paths[i]])
            for i, d in enumerate(observables_list)
        ])

    def get_final_weights(self, wc_point):
        
        struct = self.structure_constants.cpu().numpy()  # Move to CPU and convert to NumPy
        eft_weight = Event_weight_prediction1.event_weights_lin_quad(struct, wc_point)[-1]
        base = self._base_weights()
        return base * eft_weight


    def resample_observables(self, wc_point, max_events=None):
        
        if self.step == 0:
            mask = (self.final_observables['gen_l_pt'] > 0) & (self.final_observables['gen_lbar_pt'] > 0)
        elif self.step == 8:
            mask = (self.final_observables['l_pt'] > 0) & (self.final_observables['lbar_pt'] > 0)
        else:
            raise ValueError("Step must be 0 (GEN) or 8 (RECO)")
    
        data = {k: v[mask] for k, v in self.final_observables.items()}
        weights = ak.to_numpy(self.get_final_weights(wc_point)[mask]).clip(min=0)
    
        if max_events:
            subset_idx = np.random.choice(len(weights), size=min(len(weights), max_events), replace=False)
            weights = weights[subset_idx]
            data = {k: v[subset_idx] for k, v in data.items()}
    
        weights /= weights.sum()
        idx_sampled = np.random.choice(len(weights), size=len(weights), p=weights)
        sampled = {k: v[idx_sampled] for k, v in data.items()}
    
        return sampled
        