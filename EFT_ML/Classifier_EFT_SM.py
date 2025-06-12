import torch
import torch.nn as nn
import torch.optim as optim
import uproot
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class ParameterizedEFTClassifier:
    def __init__(self, variables, coeff_names, input_dim, batch_size=1024, lr=0.001):
        self.variables = variables
        self.coeff_names = coeff_names
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model(input_dim).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size

    def build_model(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.LeakyReLU(),
            #nn.Dropout(0.4),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            #nn.Dropout(0.4),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def prepare_data(self, df):
        X_obs = df[self.variables].values
        X_coeff = df[self.coeff_names].values
        X = np.hstack([X_obs, X_coeff])
        Y = df["isSignal"].values
        self.X = self.scaler.fit_transform(X)
        self.Y = Y
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, test_size=0.1, random_state=42)

    def train(self, epochs=50):
        train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.Y_train, dtype=torch.float32).view(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_data = torch.tensor(self.X_val, dtype=torch.float32).to(self.device, non_blocking=True)
        val_label = torch.tensor(self.Y_val, dtype=torch.float32).view(-1, 1).to(self.device, non_blocking=True)
        losses, val_losses = [], []

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.model.train()
            batch_loss = []

            for x, y_b in train_loader:
                x = x.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y_b)
                batch_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = np.mean(batch_loss)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_loss:.4f}")

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_data)
                val_loss = self.loss_fn(val_output, val_label)
                val_losses.append(val_loss.item())
                print(f"Epoch {epoch+1}/{epochs} - Validation loss: {val_loss.item():.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict_likelihood_ratio(self, X_input):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(self.scaler.transform(X_input), dtype=torch.float32).to(self.device)
            f_x = self.model(X_tensor).cpu().numpy().flatten()
            eps = 1e-8
            r_x = f_x / (1 - f_x + eps)
        return r_x

    def plot_llr_2d(self, eft_point, x1="mttbar", x2="cosphi", x1_range=(400,1600), x2_range=(-1,1), bins=50):
        assert x1 in self.variables and x2 in self.variables, f"{x1} and {x2} must be in variables"
        assert len(eft_point) == len(self.coeff_names), "EFT point must match number of coefficients"

        x1_vals = np.linspace(*x1_range, bins)
        x2_vals = np.linspace(*x2_range, bins)
        x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

        grid_points = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

        input_data = []
        for pt in grid_points:
            obs = np.zeros(len(self.variables))
            obs[self.variables.index(x1)] = pt[0]
            obs[self.variables.index(x2)] = pt[1]
            full_input = np.concatenate([obs, eft_point])
            input_data.append(full_input)

        input_data = np.array(input_data)
        log_r = np.log(self.predict_likelihood_ratio(input_data)).reshape(x1_grid.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(x1_grid, x2_grid, log_r, levels=50, cmap='viridis')
        plt.colorbar(contour, label=r'$\log r(x; \vec{c})$')
        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.title(f'Log-Likelihood Ratio in {x1} vs {x2} Phase Space')
        plt.tight_layout()
        plt.show()

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
            Y_val_tensor = torch.tensor(self.Y_val, dtype=torch.float32).view(-1, 1).to(self.device)

            y_pred = self.model(X_val_tensor).cpu().numpy().flatten()
            y_true = self.Y_val
            y_pred_label = (y_pred > 0.5).astype(int)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="darkorange")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred_label)
            plt.subplot(1, 2, 2)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Background", "Signal"],
                        yticklabels=["Background", "Signal"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")

            plt.tight_layout()
            plt.show()
