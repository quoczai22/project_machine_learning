"""Preprocessing functions shared by classification and clustering modules."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def clean_telco_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert known Telco fields to usable types without mutating input.

    ``TotalCharges`` blank strings are converted to ``NaN``. Median imputation
    deliberately stays in the model pipeline, where it is fit on training data
    only and therefore cannot leak information from the test set.
    """
    cleaned_data = data.copy()
    cleaned_data["TotalCharges"] = pd.to_numeric(
        cleaned_data["TotalCharges"], errors="coerce"
    )
    if cleaned_data[TARGET_COLUMN].dtype == object:
        cleaned_data[TARGET_COLUMN] = cleaned_data[TARGET_COLUMN].map(
            {"No": 0, "Yes": 1}
        )
    if cleaned_data[TARGET_COLUMN].isna().any():
        raise ValueError("Churn must contain only No/Yes or 0/1 values.")
    cleaned_data[TARGET_COLUMN] = cleaned_data[TARGET_COLUMN].astype(int)
    return cleaned_data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and binary target, excluding customer identifier."""
    required = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    features = data.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    target = data[TARGET_COLUMN].copy()
    return features, target


def build_classification_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create an unfitted train-only preprocessing transformer.

    Numeric columns receive median imputation and standard scaling. Text
    columns receive one-hot encoding. Callers must fit the returned transformer
    only through a train-set model pipeline.
    """
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
