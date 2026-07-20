import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, split_name: str = "") -> dict:
    """Print and return RMSE and R² for a given split."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{split_name} RMSE: {rmse:.4f}")
    print(f"{split_name} R²:   {r2:.4f}")
    return {"rmse": rmse, "r2": r2}


def save_model(model, filepath: str) -> None:
    """Serialize a model to disk with pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved → {filepath}")


def load_model(filepath: str):
    """Load a pickle-serialized model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
