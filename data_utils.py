from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from paths import DATASET_PATH, MODEL_PATH


REQUIRED_COLUMNS = ["date", "merchant", "description", "amount", "category", "profile", "bank", "currency"]


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    for col in ["merchant", "description", "category", "profile", "bank", "currency"]:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip()

    data = data.dropna(subset=["date", "amount"])
    data["date"] = data["date"].dt.date.astype(str)
    return data


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_transactions(df)
    dates = pd.to_datetime(data["date"], errors="coerce")
    data["month"] = dates.dt.month
    data["day_of_week"] = dates.dt.dayofweek
    data["is_income"] = (data["amount"] > 0).astype(int)
    data["abs_amount"] = data["amount"].abs()
    data["text"] = (
        data["merchant"].fillna("")
        + " "
        + data["description"].fillna("")
        + " "
        + data["profile"].fillna("")
        + " "
        + data["bank"].fillna("")
    )
    return data


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run generate_synthetic_dataset.py first.")
    return normalize_transactions(pd.read_csv(path))


def load_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def predict_categories(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    data = normalize_transactions(df)
    package = load_model(model_path)
    if package is None:
        return data

    features = add_model_features(data)
    predictions = package["pipeline"].predict(features)
    data["category"] = package["label_encoder"].inverse_transform(predictions)
    return data


def expense_view(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_transactions(df)
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.to_period("M").astype(str)
    data["expense_amount"] = data["amount"].where(data["amount"] < 0, 0).abs()
    data["income_amount"] = data["amount"].where(data["amount"] > 0, 0)
    return data
