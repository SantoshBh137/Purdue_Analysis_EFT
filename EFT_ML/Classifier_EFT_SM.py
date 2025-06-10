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

class EFTClassifier:
    def __init__(self, signal_files, background_files, variables, max_events=500000, batch_size=1024, lr=0.001):
        self.signal_files = signal_files
        self.background_files = background_files
        self.variables = variables
        self.max_events = max_events
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(len(self.variables), 100),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def load_data(self):
        def read_file(filepath, label):
            with uproot.open(filepath) as f:
                tree = f["ttBar_treeVariables_step0"]
                data = tree.arrays(self.variables + ["final_weight"], library="pd")
                data = data[(data[self.variables[0]] > -999)]
                data[self.variables] = data[self.variables].multiply(data["final_weight"], axis=0)
                data["isSignal"] = 1 if label == "signal" else 0
                data["weight"] = data["final_weight"]
                return data.drop(columns=["final_weight"])

        self.signal_dfs = [read_file(f, "signal") for f in self.signal_files.values()]
        self.bkg_dfs = [read_file(f, "background") for f in self.background_files.values()]

    def prepare_training_data(self):
        df_signal_all = pd.concat(self.signal_dfs)
        df_bkg_all = pd.concat(self.bkg_dfs)

        df_signal_all = df_signal_all.sample(n=min(self.max_events, len(df_signal_all)), random_state=42)
        df_bkg_all = df_bkg_all.sample(n=min(self.max_events, len(df_bkg_all)), random_state=42)

        df_sig_train, df_sig_test = train_test_split(df_signal_all, test_size=0.1, random_state=9)
        df_bkg_train, df_bkg_test = train_test_split(df_bkg_all, test_size=0.1, random_state=9)

        df_train = pd.concat([df_sig_train, df_bkg_train])
        self.train_weights = df_train["weight"].values
        self.X = df_train[self.variables].values
        self.Y = df_train["isSignal"].values
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_val, self.Y_train, self.Y_val = [
            np.asarray(arr) for arr in train_test_split(self.X, self.Y, test_size=0.1, random_state=9)
        ]

        self.X_sig_test = self.scaler.transform(df_sig_test[self.variables].values)
        self.Y_sig_test = df_sig_test["isSignal"].values
        self.X_bkg_test = self.scaler.transform(df_bkg_test[self.variables].values)
        self.Y_bkg_test = df_bkg_test["isSignal"].values

    def train(self, epochs=50):
        losses, val_losses = [], []

        train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.Y_train, dtype=torch.float32).view(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_data = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        val_label = torch.tensor(self.Y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.model.train()
            batch_loss = []

            for x, y_b in train_loader:
                x, y_b = x.to(self.device), y_b.to(self.device)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y_b)
                batch_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_batch_loss = np.mean(batch_loss)
            losses.append(avg_batch_loss)
            print(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_batch_loss:.4f}")

            self.model.eval()
            with torch.no_grad():
                output = self.model(val_data)
                val_loss = self.loss_fn(output, val_label)
                val_losses.append(val_loss.item())
                print(f"Epoch {epoch+1}/{epochs} - Validation loss: {val_loss.item():.4f}")

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return losses, val_losses

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            X_test = np.vstack([self.X_sig_test, self.X_bkg_test])
            Y_test = np.hstack([self.Y_sig_test, self.Y_bkg_test])

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(self.device)

            test_pred = self.model(X_test_tensor).cpu().numpy()
            test_labels = (test_pred > 0.5).astype(int)

        # ROC Curve and AUC
        fpr, tpr, _ = roc_curve(Y_test, test_pred)
        auc_score = roc_auc_score(Y_test, test_pred)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        # Confusion Matrix
        cm = confusion_matrix(Y_test, test_labels)
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Background", "Signal"],
                    yticklabels=["Background", "Signal"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        plt.tight_layout()
        plt.show()

        return Y_test, test_pred, test_labels
