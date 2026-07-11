# -*- coding: utf-8 -*-
"""Real-time prediction for NRL ↔ SL translation."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "competition_translation_model.pkl"
RESIDUALS_PATH = Path(__file__).parent / "transfer_residuals.csv"

# Load model metadata
with open(MODEL_PATH, "rb") as f:
    metadata = pickle.load(f)

model = metadata["model"]
scaler = metadata["scaler"]
features = metadata["features"]
position_map = metadata["position_map"]
residuals_std = metadata["rmse"]  # Use RMSE as confidence estimate

# Load historical residuals for position-specific confidence
residuals = pd.read_csv(RESIDUALS_PATH)
position_residuals = residuals.groupby("position")["residual"].std()

def predict_translation(player_name, position, form_z, class_z, age=26, games_per_season=17, injury_rate=0.1):
    """
    Predict NRL → SL translation factor for a player.

    Args:
        player_name: str
        position: str (e.g. "Fullback", "Halfback")
        form_z: float (form z-score in NRL)
        class_z: float (class z-score in NRL)
        age: int (approximate)
        games_per_season: float (matches per season)
        injury_rate: float (0-1, fraction of games missed)

    Returns:
        dict with prediction, confidence, interpretation
    """

    # Build feature vector
    X = np.zeros(len(features))

    # Continuous features
    X[features.index("form_z_pre")] = form_z
    X[features.index("class_z_pre")] = class_z
    X[features.index("age")] = age
    X[features.index("games_per_season")] = games_per_season
    X[features.index("injury_rate")] = injury_rate

    # One-hot position
    pos_code = position_map.get(position, "FB")  # Default to Fullback
    pos_feature = f"pos_{pos_code}"
    if pos_feature in features:
        X[features.index(pos_feature)] = 1

    # Scale and predict
    X_scaled = scaler.transform(X.reshape(1, -1))
    translation_factor = model.predict(X_scaled)[0]

    # Confidence: position-specific residual std
    position_confidence = position_residuals.get(position, residuals_std) if not isinstance(position_residuals, pd.Series) else position_residuals.get(position, residuals_std)
    if pd.isna(position_confidence):
        position_confidence = residuals_std
    confidence_band = 1.96 * position_confidence  # 95% CI

    # Interpretation
    pred_form = form_z + translation_factor
    interpretation = ""
    if translation_factor < -0.7:
        interpretation = "High translation risk (significant form drop expected)"
    elif translation_factor < -0.3:
        interpretation = "Moderate adaptation needed (some form drop)"
    elif translation_factor < 0.2:
        interpretation = "Good fit (minor adjustment expected)"
    else:
        interpretation = "Excellent fit (potential boost in SL)"

    return {
        "player_name": player_name,
        "position": position,
        "translation_factor": translation_factor,
        "predicted_form_z": pred_form,
        "confidence_lower": translation_factor - confidence_band / 2,
        "confidence_upper": translation_factor + confidence_band / 2,
        "confidence_band": confidence_band,
        "interpretation": interpretation,
        "form_drop_pct": f"{abs(translation_factor) * 10:.0f}%",  # Rough % drop
    }

# Test
if __name__ == "__main__":
    # Mock test player (Clayton Faulalo)
    result = predict_translation(
        player_name="Clayton Faulalo",
        position="Fullback",
        form_z=1.2,
        class_z=0.8,
        age=26,
        games_per_season=20,
        injury_rate=0.05
    )
    print(f"\nTranslation Prediction for {result['player_name']} ({result['position']}):")
    print(f"  Model translation factor: {result['translation_factor']:.2f}")
    print(f"  Predicted SL form z-score: {result['predicted_form_z']:.2f}")
    print(f"  Confidence band (95%): [{result['confidence_lower']:.2f}, {result['confidence_upper']:.2f}]")
    print(f"  Interpretation: {result['interpretation']}")
