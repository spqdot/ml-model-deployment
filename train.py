"""
train.py — Train Linear Regression, Random Forest, and XGBoost models,
evaluate them on train/val/test splits, and save all artefacts to models/.

Usage:
    python train.py                          # uses default DATA_PATH
    python train.py --data path/to/sales.csv # custom CSV location
"""

import os
import argparse

from src.preprocessing import (
    load_data,
    clean_data,
    split_data,
    encode_features,
    scale_features,
)
from src.model_training import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)
from src.utils import evaluate_model, save_model

MODELS_DIR = "models"
DEFAULT_DATA_PATH = "data/sales.csv"


def main(data_path: str) -> None:
    # ── Load & clean ────────────────────────────────────────────────
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    df = clean_data(df)

    # ── Split (70 / 15 / 15) ────────────────────────────────────────
    print("Splitting data  →  70% train | 15% val | 15% test")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── Encode categorical features ──────────────────────────────────
    print("One-hot encoding 'state_holiday'...")
    X_train_enc, X_val_enc, X_test_enc, encoder = encode_features(
        X_train, X_val, X_test
    )

    # ── Scale (used only for Linear Regression) ──────────────────────
    print("Applying StandardScaler...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_enc, X_val_enc, X_test_enc
    )

    # Save encoder and scaler so predict.py can reuse them
    save_model(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
    save_model(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # ── Linear Regression ───────────────────────────────────────────
    print("\n=== Linear Regression ===")
    lr = train_linear_regression(X_train_scaled, y_train)
    evaluate_model(y_train, lr.predict(X_train_scaled), "Train      ")
    evaluate_model(y_val,   lr.predict(X_val_scaled),   "Validation ")
    evaluate_model(y_test,  lr.predict(X_test_scaled),  "Test       ")
    save_model(lr, os.path.join(MODELS_DIR, "linear_regression.pkl"))

    # ── Random Forest ───────────────────────────────────────────────
    print("\n=== Random Forest ===")
    rf = train_random_forest(X_train_enc, y_train)
    evaluate_model(y_train, rf.predict(X_train_enc), "Train      ")
    evaluate_model(y_val,   rf.predict(X_val_enc),   "Validation ")
    evaluate_model(y_test,  rf.predict(X_test_enc),  "Test       ")
    save_model(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    # ── XGBoost ─────────────────────────────────────────────────────
    print("\n=== XGBoost ===")
    xgb = train_xgboost(X_train_enc, y_train)
    evaluate_model(y_train, xgb.predict(X_train_enc), "Train      ")
    evaluate_model(y_val,   xgb.predict(X_val_enc),   "Validation ")
    evaluate_model(y_test,  xgb.predict(X_test_enc),  "Test       ")
    save_model(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))

    print("\nAll models trained and saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models for sales prediction")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the sales CSV file (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()
    main(args.data)
