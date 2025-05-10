import torch
from torch.utils.data import Dataset

class WilsonDataset(Dataset):
    def __init__(self, observables, event_weights, labels):
        """
        Initializes the dataset.
        
        Parameters:
        - observables: List or array of observables (e.g., `file['c_kk']`).
        - event_weights: List or array of weights corresponding to each event (e.g., `w1`, `w1_0`).
        - labels: List or array of labels (e.g., SM (0) or EFT (1)).
        """
        self.observables = torch.tensor(observables, dtype=torch.float32)  # Observables (e.g., 'c_kk')
        self.event_weights = torch.tensor(event_weights, dtype=torch.float32).unsqueeze(1)  # Event weights
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Labels (SM or EFT)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Reweight the observables by multiplying with event weight
        x = self.observables[idx] * self.event_weights[idx]  # Apply weight to the observable
        return x, self.labels[idx], self.event_weights[idx]  # Return reweighted feature, label, and event weight
