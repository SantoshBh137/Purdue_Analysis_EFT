from pathlib import Path
import os
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/depot/cms/top/bhanda25/Purdue_Analysis_EFT/Purdue_Analysis_EFT/EFT_minitree')
import Event_weight_prediction1


class EFTReweighter:
    def __init__(self, directory_path, eras, channels, mass_regions, cross_sections, struct_const_dir):
        self.directory_path = directory_path
        self.cross_sections = cross_sections
        self.struct_const_dir = struct_const_dir

        self.file_paths = [
            os.path.join(directory_path,
                         f"spinCorrInput_{era}/Nominal/{channel}/{channel}_ttto2l2nu_jet_smeft_mtt_{region}_{era}.root")
            for era in eras
            for channel in channels
            for region in mass_regions
        ]

        self.struct_paths_gen = [
            os.path.join(struct_const_dir,
                         f"saved_sc_{era}/Nominal/{channel}_ttto2l2nu_jet_smeft_mtt_{region}_{era}_struct_gen.npy")
            for era in eras
            for channel in channels
            for region in mass_regions
        ]

        self.struct_paths_reco = [
            os.path.join(struct_const_dir,
                         f"saved_sc_{era}/Nominal/{channel}_ttto2l2nu_jet_smeft_mtt_{region}_{era}_struct_reco.npy")
            for era in eras
            for channel in channels
            for region in mass_regions
        ]

        self.weights_step0, self.weights_step8 = self._compute_file_weights()
        self.final_observables_gen = {}
        self.final_observables_reco = {}
        self.true_weights_gen = None
        self.event_weights_reco = None
        self.structure_constants_gen = None
        self.structure_constants_reco = None

    def _compute_file_weights(self):
        weights_step0, weights_step8 = {}, {}
        grouped = {k: [] for k in self.cross_sections.keys()}
        for path in self.file_paths:
            for region in grouped:
                if region in path:
                    grouped[region].append(path)

        for region, files in grouped.items():
            total_events_step0, total_events_step8 = 0, 0
            for f in files:
                with uproot.open(f) as file:
                    total_events_step0 += file["ttBar_treeVariables_step0"].num_entries
                    total_events_step8 += file["ttBar_treeVariables_step8"].num_entries
            for f in files:
                weights_step0[f] = self.cross_sections[region] / total_events_step0 if total_events_step0 > 0 else 0
                weights_step8[f] = self.cross_sections[region] / total_events_step8 if total_events_step8 > 0 else 0

        return weights_step0, weights_step8

    def load_structure_constants(self):
        def load_constants(paths):
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
            for path in tqdm(valid_paths, desc=f"Loading {paths[0].split('_')[-1]} structure constants"):
                data = np.load(path)
                n = data.shape[0]
                struct_array[current_index:current_index + n] = data
                current_index += n

            return struct_array

        self.structure_constants_gen = load_constants(self.struct_paths_gen)
        self.structure_constants_reco = load_constants(self.struct_paths_reco)
        print("Structure constants loaded:", self.structure_constants_gen.shape, self.structure_constants_reco.shape)

    def load_observables(self):
        collected_gen, collected_reco = [], []
        true_weights_gen, event_weights_reco = [], []

        for file_path in tqdm(self.file_paths, desc="Loading observables", unit="file"):
            with uproot.open(file_path) as file:
                # GEN
                gen_tree = file["ttBar_treeVariables_step0"]
                gen_keys = [k for k in gen_tree.keys() if k.startswith("gen_")]
                arrays = gen_tree.arrays(gen_keys + ["trueLevelWeight"])
                collected_gen.append({k: arrays[k] for k in gen_keys})
                true_weights_gen.append(arrays["trueLevelWeight"])

                # RECO
                reco_tree = file["ttBar_treeVariables_step8"]
                reco_keys = [k for k in reco_tree.keys() if not k.startswith("gen_")]
                arrays = reco_tree.arrays(reco_keys + ["eventWeight"])
                collected_reco.append({k: arrays[k] for k in reco_keys})
                event_weights_reco.append(arrays["eventWeight"])

        self.collected_gen = collected_gen
        self.collected_reco = collected_reco

        self.final_observables_gen = {
            k: ak.concatenate([obs[k] for obs in collected_gen]) for k in collected_gen[0].keys()
        }
        self.true_weights_gen = ak.concatenate(true_weights_gen)

        self.final_observables_reco = {
            k: ak.concatenate([obs[k] for obs in collected_reco]) for k in collected_reco[0].keys()
        }
        self.event_weights_reco = ak.concatenate(event_weights_reco)

    def _base_weights(self, observables_list, file_weight_dict):
        example_key = next(iter(observables_list[0]))
        return ak.concatenate([
            np.full(len(d[example_key]), file_weight_dict[self.file_paths[i]])
            for i, d in enumerate(observables_list)
        ])

    def get_final_weights(self, wc_point, step):
        if step == 0:
            eft_weight = Event_weight_prediction1.event_weights_lin_quad(self.structure_constants_gen, wc_point)[-1]
            base = self._base_weights(self.collected_gen, self.weights_step0)
            return base * self.true_weights_gen * eft_weight
        elif step == 8:
            eft_weight = Event_weight_prediction1.event_weights_lin_quad(self.structure_constants_reco, wc_point)[-1]
            base = self._base_weights(self.collected_reco, self.weights_step8)
            return base * self.event_weights_reco * eft_weight
        else:
            raise ValueError("step must be 0 or 8")

    def apply_mask_and_save(self, wc_point, output_file):
        print("Applying GEN and RECO masks...")
        mask_gen = (self.final_observables_gen['gen_l_pt'] > 0) & (self.final_observables_gen['gen_lbar_pt'] > 0)
        mask_reco = (self.final_observables_reco['l_pt'] > 0) & (self.final_observables_reco['lbar_pt'] > 0)

        def convert_array(arr):
            try:
                return ak.to_numpy(arr)
            except Exception:
                return arr
        
       # Explicitly exclude weight-related branches
        excluded_gen_keys = {"trueLevelWeight", "mgWeights"}
        excluded_reco_keys = {"eventWeight", "mgWeights"}
    
        masked_gen = {
            k: v[mask_gen] for k, v in self.final_observables_gen.items()
            if k not in excluded_gen_keys
        }
        masked_reco = {
            k: v[mask_reco] for k, v in self.final_observables_reco.items()
            if k not in excluded_reco_keys
        }
        
        print("Computing weights for GEN...")
        weights_gen = self.get_final_weights(wc_point, step=0)[mask_gen]
        print("Computing weights for RECO...")
        weights_reco = self.get_final_weights(wc_point, step=8)[mask_reco]
        
        print(f"Saving masked arrays and weights to {output_file}...")
        with uproot.recreate(output_file) as f:
            masked_gen['final_weight'] = ak.to_numpy(weights_gen)
            masked_reco['final_weight'] = ak.to_numpy(weights_reco)
            f["ttBar_treeVariables_step0"] = {k: convert_array(v) for k, v in masked_gen.items()}
            f["ttBar_treeVariables_step8"] = {k: convert_array(v) for k, v in masked_reco.items()}
