"""Tests enforcing use of the paired preprocessing outputs from member 2."""

import json
from pathlib import Path

from src.experiments import load_preprocessed_splits


BASE_DIR = Path(__file__).resolve().parent.parent


def test_preprocessed_train_test_pair_is_complete_and_aligned():
    """The fixed 80/20 pair must be used directly, never split a second time."""
    train_path = BASE_DIR / "data" / "processed" / "Train_Data.csv"
    test_path = BASE_DIR / "data" / "processed" / "Test_Data.csv"
    X_train, X_test, y_train, y_test = load_preprocessed_splits(
        str(train_path), str(test_path)
    )

    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert len(X_train) + len(X_test) == 7043
    assert list(X_train.columns) == list(X_test.columns)
    assert "customerID" not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_split_metadata_records_leakage_safe_provenance():
    """The exported pair must document its split and fit-only-on-train rules."""
    metadata_path = BASE_DIR / "data" / "processed" / "classification_split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_rows"] == 7043
    assert metadata["train_rows"] == 5634
    assert metadata["test_rows"] == 1409
    assert metadata["random_state"] == 42
    assert metadata["stratify"] == "Churn"
    assert metadata["scaler_fit_on"] == "Train_Data only"
    assert metadata["encoder_fit_on"] == "Train_Data only"
