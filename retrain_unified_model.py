from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils import add_model_features, load_dataset
from paths import DATASET_PATH, MODEL_PATH


TEXT_FEATURE = "text"
NUMERIC_FEATURES = ["amount", "abs_amount", "month", "day_of_week", "is_income"]
CATEGORICAL_FEATURES = ["profile", "bank", "currency"]


def train_model(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> dict[str, object]:
    df = add_model_features(load_dataset(dataset_path))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    X = df[[TEXT_FEATURE, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]

    stratify = y if min(__import__("collections").Counter(y).values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    package = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "text_feature": TEXT_FEATURE,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classification_report": report,
    }
    joblib.dump(package, model_path)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified transaction category model.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    package = train_model(args.dataset, args.model)
    print(f"Saved model to {args.model}")
    print(package["classification_report"])


if __name__ == "__main__":
    main()
