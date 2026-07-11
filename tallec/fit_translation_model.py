# -*- coding: utf-8 -*-
"""Fit Ridge regression model for NRL ↔ SL competition translation."""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Load transfer dataset
transfers = pd.read_csv(Path(__file__).parent / "transfer_dataset.csv")
print(f"Loaded {len(transfers)} historical transfers")

# Filter: only complete data
transfers = transfers.dropna(subset=["form_z_pre", "class_z_pre", "form_z_post", "translation_factor"])
print(f"Using {len(transfers)} complete transfers for training")

# Position categories
position_map = {
    "Fullback": "FB",
    "Wing": "W",
    "Centre": "C",
    "Five-Eighth": "H5",
    "Halfback": "HB",
    "Prop": "P",
    "2nd Row": "2R",
    "Lock": "LK",
    "Hooker": "H",
}
transfers["pos_code"] = transfers["position"].map(position_map)

# Features: form_z, class_z, age, games/season, injury_rate, position (one-hot)
X = transfers[["form_z_pre", "class_z_pre", "age", "games_per_season", "injury_rate"]].copy()

# One-hot encode position
position_dummies = pd.get_dummies(transfers["pos_code"], prefix="pos")
X = pd.concat([X, position_dummies], axis=1)

# Target: translation_factor (expected drop/boost)
y = transfers["translation_factor"]

print(f"\nFeature matrix: {X.shape}")
print(f"Target (translation_factor) stats:\n{y.describe()}")

# Train/test split: last 5 as holdout
X_train, X_test = X[:-5], X[-5:]
y_train, y_test = y[:-5], y[-5:]

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Ridge (alpha=1.0, moderate regularization)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\nModel Performance:")
print(f"  Train R2: {train_score:.3f}")
print(f"  Test R2 (holdout 5): {test_score:.3f}")

# Predictions on test set
y_pred = model.predict(X_test_scaled)
residuals = y_test.values - y_pred
rmse = np.sqrt(np.mean(residuals**2))
print(f"  Test RMSE: {rmse:.3f}")

print(f"\nHoldout Test Predictions vs Actual:")
for i, (idx, row) in enumerate(transfers[-5:].iterrows()):
    print(f"  {row['player_name']:25s} ({row['position']:12s}): actual {y_test.iloc[i]:6.2f}, pred {y_pred[i]:6.2f}, error {residuals[i]:6.2f}")

# Feature importance (coefficients)
feature_names = list(X.columns)
coefs = pd.DataFrame({"feature": feature_names, "coef": model.coef_}).sort_values("coef", key=abs, ascending=False)
print(f"\nTop Feature Importances:")
print(coefs.head(10).to_string(index=False))

# Save model
model_path = Path(__file__).parent / "competition_translation_model.pkl"
metadata = {
    "model": model,
    "scaler": scaler,
    "features": feature_names,
    "position_map": position_map,
    "rmse": rmse,
    "test_r2": test_score,
    "position_codes": list(position_dummies.columns),
}
with open(model_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"\nModel saved to {model_path}")

# Save residuals for confidence calculation
residuals_df = pd.DataFrame({
    "player": transfers[-5:]["player_name"].values,
    "position": transfers[-5:]["position"].values,
    "residual": residuals if isinstance(residuals, np.ndarray) else residuals.values,
})
residuals_df.to_csv(Path(__file__).parent / "transfer_residuals.csv", index=False)
print(f"Residuals saved for confidence scoring")
