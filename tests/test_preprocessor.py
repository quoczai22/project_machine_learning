"""Unit tests for Telco preprocessing."""

import pandas as pd

from src.preprocessor import (
    build_classification_preprocessor,
    clean_telco_data,
    split_features_target,
)


def sample_raw_data() -> pd.DataFrame:
    """Provide a minimal Telco-like table including a blank TotalCharges value."""
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "TotalCharges": [" ", "20.5", "70.0"],
            "tenure": [0, 2, 7],
            "Partner": ["Yes", "No", "No"],
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_clean_telco_data_converts_total_charges_and_target() -> None:
    """Blank TotalCharges becomes NaN and target becomes binary."""
    cleaned = clean_telco_data(sample_raw_data())

    assert cleaned["TotalCharges"].isna().sum() == 1
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["Churn"].tolist() == [1, 0, 0]


def test_split_features_target_excludes_id_and_target() -> None:
    """Identifiers and label must never become classification features."""
    features, target = split_features_target(clean_telco_data(sample_raw_data()))

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert target.tolist() == [1, 0, 0]


def test_preprocessor_imputes_and_encodes() -> None:
    """The train-only transformer handles missing numeric and text columns."""
    features, _ = split_features_target(clean_telco_data(sample_raw_data()))
    transformed = build_classification_preprocessor(features).fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert pd.isna(transformed.toarray() if hasattr(transformed, "toarray") else transformed).sum() == 0
