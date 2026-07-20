# ML Model Deployment

Sales prediction service using **Linear Regression**, **Random Forest**, and **XGBoost**, served via a **FastAPI** REST API.

---

## Project Structure

```
ml_model_deployment/
│
├── app.py              # FastAPI application
├── train.py            # Train all three models
├── predict.py          # Prediction helper functions
├── requirements.txt    # Python dependencies
│
├── data/
│   └── sales.csv       # ← Place your training CSV here
│
├── models/             # Saved model artefacts (generated after training)
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
└── src/
    ├── preprocessing.py    # Data loading, cleaning, encoding, scaling
    ├── model_training.py   # Model factory functions
    └── utils.py            # Evaluation, save/load helpers
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train models

Place your `sales.csv` file in the `data/` folder, then run:

```bash
python train.py
# or specify a custom path:
python train.py --data path/to/your/sales.csv
```

Expected CSV columns: `store_ID`, `open`, `promo`, `state_holiday`, `school_holiday`, `sales`
(columns `Unnamed: 0` and `date` are dropped automatically if present).

This trains all three models and saves artefacts to `models/`.

---

### Predict on new data (CLI)

```bash
python predict.py --data path/to/new_data.csv --model xgboost --output predicted_sales.csv
```

Available `--model` values: `linear_regression`, `random_forest`, `xgboost`

---

### Start the API server

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs: `http://127.0.0.1:8000/docs`

---

### API Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Health check                       |
| GET    | `/models`  | List available model names         |
| POST   | `/predict` | Predict sales for one or more rows |

#### POST `/predict` — example request body

```json
{
  "records": [
    {
      "store_ID": 1,
      "open": 1,
      "promo": 1,
      "state_holiday": "0",
      "school_holiday": 0
    }
  ],
  "model_name": "xgboost"
}
```

#### Response

```json
{
  "predictions": [5243.17],
  "model_used": "xgboost",
  "num_records": 1
}
```

---

## Model Performance Summary

| Model             | Notes                                              |
|-------------------|----------------------------------------------------|
| Linear Regression | Baseline; uses StandardScaler                      |
| Random Forest     | Ensemble of decision trees; good generalisation    |
| XGBoost           | Best performing model; recommended for production  |
