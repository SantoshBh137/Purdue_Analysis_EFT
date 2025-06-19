import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data
def load_data(signal_path, background_path, VARS, max_events):
    # All base variables needed, plus any that are used in derived features
    derived_vars = [
        'gen_crk_ckr', 'gen_crk_ckr_m',
        'gen_cnr_crn', 'gen_cnr_crn_m',
        'gen_cnk_ckn', 'gen_cnk_ckn_m'
    ]
    base_vars = list(set(VARS) - set(derived_vars))
    base_vars += ['gen_c_rk', 'gen_c_kr', 'gen_c_nr', 'gen_c_rn', 'gen_c_nk', 'gen_c_kn']

    # Load only necessary base variables from "gen"
    df_sig = pd.read_pickle(signal_path)["gen"][base_vars].copy()
    df_bkg = pd.read_pickle(background_path)["gen"][base_vars].copy()

    # Add derived variables
    for df in [df_sig, df_bkg]:
        df["gen_crk_ckr"]    = df["gen_c_rk"] + df["gen_c_kr"]
        df["gen_crk_ckr_m"]  = df["gen_c_rk"] - df["gen_c_kr"]
        df["gen_cnr_crn"]    = df["gen_c_nr"] + df["gen_c_rn"]
        df["gen_cnr_crn_m"]  = df["gen_c_nr"] - df["gen_c_rn"]
        df["gen_cnk_ckn"]    = df["gen_c_nk"] + df["gen_c_kn"]
        df["gen_cnk_ckn_m"]  = df["gen_c_nk"] - df["gen_c_kn"]

    # Sample
    df_sig = df_sig.sample(n=min(max_events, len(df_sig)), random_state=9)
    df_bkg = df_bkg.sample(n=min(max_events, len(df_bkg)), random_state=9)

    # Labels
    df_sig["isSignal"] = 1
    df_bkg["isSignal"] = 0

    # Split
    df_sig_train, df_sig_test = train_test_split(df_sig, test_size=0.1, random_state=9)
    df_bkg_train, df_bkg_test = train_test_split(df_bkg, test_size=0.1, random_state=9)

    df_train = pd.concat([df_sig_train, df_bkg_train])
    df_test_sig = pd.concat([df_sig_test])
    df_test_bkg = pd.concat([df_bkg_test])
    df_combined = pd.concat([df_sig, df_bkg])

    # Extract inputs and labels
    X = df_train[VARS].values
    Y = df_train["isSignal"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.1, random_state=9)

    X_sig_test = scaler.transform(df_test_sig[VARS].values)
    X_bkg_test = scaler.transform(df_test_bkg[VARS].values)

    return X_train, Y_train, X_val, Y_val, X_sig_test, X_bkg_test, scaler, df_combined



# Define model
class EFTClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EFTClassifier, self).__init__()
        layers = []
        for _ in range(3):
            layers += [
                nn.Linear(input_dim, 100),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Dropout(0.1)
            ]
            input_dim = 100
        layers.append(nn.Linear(100, 1))  # No sigmoid
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training
def train_model(X_train, Y_train, X_val, Y_val, input_dim, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EFTClassifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 8192
    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).float().unsqueeze(1)
    X_val_tensor = torch.from_numpy(X_val).float()
    Y_val_tensor = torch.from_numpy(Y_val).float().unsqueeze(1)

    train_ds = TensorDataset(X_train_tensor, Y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    best_val_loss = float('inf')
    patience, epochs_no_improve = 10, 0
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    return model, device

# Likelihood ratio plot
def plot_llr_2d(model, scaler, vars_all, x_var, y_var, x_range, y_range, bins=60):
    xi, yi = vars_all.index(x_var), vars_all.index(y_var)
    xs = np.linspace(*x_range, bins)
    ys = np.linspace(*y_range, bins)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    Xgrid = np.zeros((len(grid), len(vars_all)))
    Xgrid[:, xi], Xgrid[:, yi] = grid[:, 0], grid[:, 1]
    Xgrid_scaled = scaler.transform(Xgrid)

    inputs = torch.tensor(Xgrid_scaled, dtype=torch.float32).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(inputs)).cpu().numpy().flatten()

    r = scores / (1 - scores + 1e-9)
    log_r = np.log(r).reshape(bins, bins)
    plt.figure(figsize=(8, 6))
    plt.imshow(log_r.T, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]],
               aspect='auto', cmap='bwr_r', vmin=-0.4, vmax=0.4)
    plt.colorbar(label=r'$\log r(x|\mathrm{SM},\; c_{tG} = 2)$')
    plt.xlabel(r'$\cos(\phi)$')
    plt.ylabel(r'$m_{t\bar{t}}$')
    plt.tight_layout()
    plt.show()
    return scores, log_r

# Score distribution plot
def plot_scores(model, device, X_sig_test, X_bkg_test):
    model.eval()
    with torch.no_grad():
        sig_logits = model(torch.tensor(X_sig_test, dtype=torch.float32).to(device))
        bkg_logits = model(torch.tensor(X_bkg_test, dtype=torch.float32).to(device))

        sig_scores = torch.sigmoid(sig_logits).cpu().numpy().flatten()
        bkg_scores = torch.sigmoid(bkg_logits).cpu().numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(sig_scores, bins=50, alpha=0.3, color='blue', label='EFT', density=True)
    plt.hist(bkg_scores, bins=50, alpha=0.3, color='red', label='SM', density=True)
    plt.xlabel("DNN Score")
    plt.ylabel("Density")
    #plt.yscale("log")
    plt.title("DNN Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return sig_scores

# Histogram-based LLR
def compute_histogram_llr(x_obs, y_obs, label_obs, x_range, y_range, bins=60):
    x_bins = np.linspace(*x_range, bins + 1)
    y_bins = np.linspace(*y_range, bins + 1)

    x_obs = np.asarray(x_obs)
    y_obs = np.asarray(y_obs)
    label_obs = np.asarray(label_obs)

    sm_mask = label_obs == 0
    eft_mask = label_obs == 1

    hist_sm, _, _ = np.histogram2d(x_obs[sm_mask], y_obs[sm_mask], bins=[x_bins, y_bins], density=True)
    hist_eft, _, _ = np.histogram2d(x_obs[eft_mask], y_obs[eft_mask], bins=[x_bins, y_bins], density=True)

    epsilon = 1e-9
    ratio = np.log((hist_sm + epsilon) / (hist_eft + epsilon))

    plt.figure(figsize=(8, 6))
    plt.imshow(ratio.T, origin='lower', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
               aspect='auto', cmap='bwr_r', vmin=-0.4, vmax=0.4)
    plt.colorbar(label=r'$\log r(x|\mathrm{SM},\; c_{tG}=2)$')
    plt.xlabel(r'$\cos(\phi)$')
    plt.ylabel(r'$m_{t\bar{t}}$ [GeV]')
    plt.title('Log-Likelihood Ratio (Histogram)')
    plt.tight_layout()
    plt.show()

    return ratio

# Evaluation
def evaluate_model(model, device, X_sig_test, X_bkg_test):
    model.eval()
    with torch.no_grad():
        X_sig_tensor = torch.tensor(X_sig_test, dtype=torch.float32).to(device)
        X_bkg_tensor = torch.tensor(X_bkg_test, dtype=torch.float32).to(device)

        sig_scores = torch.sigmoid(model(X_sig_tensor)).cpu().numpy().flatten()
        bkg_scores = torch.sigmoid(model(X_bkg_tensor)).cpu().numpy().flatten()

    scores = np.concatenate([sig_scores, bkg_scores])
    labels = np.concatenate([np.ones_like(sig_scores), np.zeros_like(bkg_scores)])

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    preds_binary = (scores > 0.5).astype(int)
    cm = confusion_matrix(labels, preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.show()

    return fpr, tpr, roc_auc, cm
