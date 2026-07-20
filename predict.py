"""
predict.py — Load saved models and run predictions on new data.

Usage (standalone):
    python predict.py --data path/to/new_data.csv --model xgboost
"""

import os
import argparse

import pandas as pd

from src.preprocessing import clean_data, preprocess_new_data
from src.utils import load_model

MODELS_DIR = "models"
MODEL_NAMES = ("linear_regression", "random_forest", "xgboost")


def _load_artifacts():
    """Load encoder, scaler, and all three trained models."""
    encoder = load_model(os.path.join(MODELS_DIR, "encoder.pkl"))
    scaler = load_model(os.path.join(MODELS_DIR, "scaler.pkl"))
    models = {
        name: load_model(os.path.join(MODELS_DIR, f"{name}.pkl"))
        for name in MODEL_NAMES
    }
    return encoder, scaler, models


def predict(new_data: pd.DataFrame, model_name: str = "xgboost") -> list:
    """
    Preprocess new_data and return predictions from the specified model.

    Parameters
    ----------
    new_data   : DataFrame with the same feature columns as training data
                 (may include 'Unnamed: 0' / 'date' / 'index' — they are dropped).
    model_name : one of 'linear_regression', 'random_forest', 'xgboost'

    Returns
    -------
    list of float predictions
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {MODEL_NAMES}"
        )

    encoder, scaler, models = _load_artifacts()

    df = clean_data(new_data.copy())

    # Linear Regression was trained on scaled features; tree models were not.
    use_scaler = scaler if model_name == "linear_regression" else None
    X_processed = preprocess_new_data(df, encoder, scaler=use_scaler)

    predictions = models[model_name].predict(X_processed)
    return predictions.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on new data")
    parser.add_argument("--data",  required=True, help="Path to new data CSV")
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=MODEL_NAMES,
        help="Model to use for prediction (default: xgboost)",
    )
    parser.add_argument(
        "--output",
        default="predicted_sales.csv",
        help="Output CSV file (default: predicted_sales.csv)",
    )
    args = parser.parse_args()

    df_new = pd.read_csv(args.data)
    preds = predict(df_new, model_name=args.model)

    df_new["sales"] = preds
    df_new.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
