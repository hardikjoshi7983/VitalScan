
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.train_models import make_synthetic_data, DISEASES
from src.features import FEATURES, NUMERIC

class TabularTransformer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.projection = nn.Linear(n_features, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           dropout=.15, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in DISEASES])
    def forward(self, x):
        tokens = self.projection(x).unsqueeze(1)
        encoded = self.encoder(tokens).mean(dim=1)
        return torch.cat([head(encoded) for head in self.heads], dim=1)

def main():
    torch.manual_seed(42)
    df = make_synthetic_data(6500)
    # Numeric-only baseline for a compact Transformer experiment; categorical values encoded explicitly.
    X = df[NUMERIC].copy()
    X["sex_male"] = (df.sex == "Male").astype(float)
    X["sex_other"] = (df.sex == "Other").astype(float)
    X["smoking_yes"] = (df.smoking == "Yes").astype(float)
    X["family_yes"] = (df.family_history == "Yes").astype(float)
    X = X.fillna(X.median())
    feature_names = list(X.columns)  # Keep names before converting the DataFrame to NumPy.
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")
    y = df[DISEASES].to_numpy(dtype="float32")
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=.22, random_state=42)
    model = TabularTransformer(X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    tx, ty = torch.tensor(xtr), torch.tensor(ytr)
    for epoch in range(35):
        model.train()
        order = torch.randperm(len(tx))
        for idx in order.split(128):
            optimizer.zero_grad()
            loss = loss_fn(model(tx[idx]), ty[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(xte))).numpy()
    print("Transformer ROC-AUC (synthetic labels):")
    for i, disease in enumerate(DISEASES):
        print(f"  {disease}: {roc_auc_score(yte[:, i], probs[:, i]):.3f}")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": X.shape[1],
                "feature_names": feature_names}, "artifacts/tabular_transformer.pt")
    print("Saved artifacts/tabular_transformer.pt")

if __name__ == "__main__":
    main()
