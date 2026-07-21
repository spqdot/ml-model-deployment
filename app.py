"""
app.py — FastAPI application exposing prediction endpoints.

Start the server:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import pandas as pd

from predict import predict, MODEL_NAMES

app = FastAPI(
    title="ML Model Deployment API",
    description="Predict sales using Linear Regression, Random Forest, or XGBoost.",
    version="1.0.0",
)

# Allow requests from any origin (update to your Vercel URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Request / Response schemas ──────────────────────────────────────────────

class SaleRecord(BaseModel):
    store_ID: int
    day_of_week: int
    nb_customers_on_day: int
    open: int
    promotion: int
    state_holiday: str
    school_holiday: int


class PredictionRequest(BaseModel):
    records: List[SaleRecord]
    model_name: Optional[str] = "xgboost"


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_used: str
    num_records: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "ML Model Deployment API is running"}


@app.get("/models", tags=["Info"])
def list_models():
    """Return the list of available model names."""
    return {"available_models": list(MODEL_NAMES)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def make_prediction(request: PredictionRequest):
    """
    Accept one or more sale records and return predictions.

    - **records**: list of feature objects (store_ID, day_of_week, nb_customers_on_day, open, promotion, state_holiday, school_holiday)
    - **model_name**: `linear_regression` | `random_forest` | `xgboost` (default: `xgboost`)
    """
    try:
        df = pd.DataFrame([r.model_dump() for r in request.records])
        preds = predict(df, model_name=request.model_name)
        return PredictionResponse(
            predictions=preds,
            model_used=request.model_name,
            num_records=len(preds),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model files not found. Run `python train.py` first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
