"""Regression checks for the committed processed training dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_TRAIN = PROJECT_ROOT / "data" / "processed" / "telco_train_data.csv"


def test_processed_train_data_has_expected_schema_and_values() -> None:
    """Processed training data remains complete and ready for model training."""
    data = pd.read_csv(PROCESSED_TRAIN)

    assert data.shape == (5634, 32)
    assert data.isna().sum().sum() == 0
    assert set(data["Churn"].unique()) == {0, 1}
    assert pd.api.types.is_numeric_dtype(data["TotalCharges"])
    assert "customerID" in data.columns
