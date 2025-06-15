# eft_likelihood_calibrated_gen_only.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt



# ——————————————
# 3. Classifier model
# ——————————————
class ParameterizedClassifier:
    def __init__(self, n_input, lr=1e-2, batch_size=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = nn.Sequential(
            nn.Linear(n_input, 20), nn.LeakyReLU(),
            nn.Linear(20, 10), nn.LeakyReLU(),
            nn.Linear(10, 1), nn.Sigmoid()
        ).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size

    def prepare_data(self, df, gen_vars, coeff_names):
        X = np.hstack([df[gen_vars].values, df[coeff_names].values])
        Y = df["isSignal"].values

        X = self.scaler.fit_transform(X)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(X, Y, test_size=0.1, random_state=42)

    def train(self, epochs=50):
        ds = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.Y_train, dtype=torch.float32).unsqueeze(1)
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        val_X = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        val_Y = torch.tensor(self.Y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        train_losses, val_losses = [], []

        for epoch in range(1, epochs + 1):
            self.model.train()
            batch_losses = []
            for Xb, Yb in loader:
                Xb, Yb = Xb.to(self.device), Yb.to(self.device)
                pred = self.model(Xb)
                loss = self.loss_fn(pred, Yb)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                batch_losses.append(loss.item())

            avg_train = np.mean(batch_losses)
            train_losses.append(avg_train)

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(val_X)
                val_loss = self.loss_fn(pred_val, val_Y)
                avg_val = val_loss.mean().item()
                val_losses.append(avg_val)

            print(f"Epoch {epoch}/{epochs} — train {avg_train:.7f}, val {avg_val:.7f}")

        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32).to(self.device)
            return self.model(Xt).cpu().numpy().ravel()

# ——————————————
# 4. Calibrate
# ——————————————
def calibrate_outputs(clf, X_val, Y_val):
    proba = clf.predict_proba(X_val)
    Y_val = np.array(Y_val).ravel()
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(proba, Y_val)
    return ir

# ——————————————
# 5. Plot GEN-level likelihood ratio
# ——————————————
def plot_llr_2d(clf, calibrator, vars_all, coeff_names,
                eft_point, x_var, y_var, x_range, y_range, bins=60):
    xi = vars_all.index(x_var); yi = vars_all.index(y_var)
    xs = np.linspace(*x_range, bins); ys = np.linspace(*y_range, bins)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    Xgrid = np.zeros((len(grid), len(vars_all)))
    Xgrid[:, xi], Xgrid[:, yi] = grid[:, 0], grid[:, 1]
    inputs = np.hstack([Xgrid, np.tile(eft_point, (len(Xgrid),1))])

    s = clf.predict_proba(inputs)
    s = calibrator.predict(s)
    r = s / (1 - s + 1e-9)

    plt.contourf(xs, ys, np.log(r).reshape(bins, bins), levels=50, cmap="coolwarm")
    plt.colorbar(label=r'$\ln r(x)$')
    plt.xlabel(x_var); plt.ylabel(y_var)
    plt.title(f"GEN-level Log-LR @{coeff_names[0]}={eft_point[0]}")
    plt.tight_layout(); plt.show()
