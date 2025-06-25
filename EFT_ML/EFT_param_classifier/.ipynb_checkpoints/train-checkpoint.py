import numpy as np
import torch, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch import optim, nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluator import EFTReweighter
from model import ParametricClassifier
import os

# === Dataset Building ===
def build_dataset(obs0, obs1, th0, th1, var_names):
    X0 = np.stack([np.asarray(obs0[k]) for k in var_names], axis=1)
    theta0_vals = np.full((len(X0), 1), th0[0])
    X1 = np.stack([np.asarray(obs1[k]) for k in var_names], axis=1)
    theta1_vals = np.full((len(X1), 1), th1[0])
    X = np.vstack([np.hstack([X0, theta0_vals]),
                   np.hstack([X1, theta1_vals])])
    Y = np.concatenate([np.ones(len(X0)), np.zeros(len(X1))])
    return X, Y

def generate_data(reweighter, var_names, wc_dim, N, M, theta1):
    values = np.linspace(-2, 2, N)
    for v in values:
        th0 = np.zeros(wc_dim)
        th0[0] = v
        obs0 = reweighter.resample_observables(th0, M)
        obs1 = reweighter.resample_observables(theta1, M)
        yield build_dataset(obs0, obs1, th0, theta1, var_names)
        
def train(X, Y,VARS, device):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    model = ParametricClassifier(len(VARS) + 1).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    def to_tensor(x, y):
        return DataLoader(TensorDataset(torch.tensor(x).float(), torch.tensor(y).float().unsqueeze(1)), batch_size=1024, shuffle=True)

    train_loader = to_tensor(X_train, Y_train)
    val_loader = to_tensor(X_val, Y_val)

    best_loss, best_state = float('inf'), None
    train_losses, val_losses = [], []
    for epoch in range(20):
        model.train(); train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/20", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb); val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        if val_loss < best_loss: best_loss = val_loss; best_state = model.state_dict()

    model.load_state_dict(best_state)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    torch.save(model.state_dict(), "best.pt")
    joblib.dump(scaler, "scaler.pkl")
    return model, scaler, (X_test, Y_test)


def evaluate(model, scaler, X_test, Y_test, device):
    X_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    plt.hist(preds[Y_test==1], bins=50, alpha=0.6, label='Signal (EFT)', density=True)
    plt.hist(preds[Y_test==0], bins=50, alpha=0.4, label='Background (SM)', density=True, color='gray')
    plt.xlabel("Classifier Output"); plt.ylabel("Density")
    plt.title("Score Distribution on Test Set")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

