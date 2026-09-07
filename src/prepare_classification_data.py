"""Create reproducible, leakage-safe classification train/test CSV files.

The preprocessing transformers are fitted on the training split only.  The
test split is transformed with those fitted transformers and is never used to
learn medians, scaling statistics, or categorical levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
TRAIN_PATH = Path("data/processed/Train_Data.csv")
TEST_PATH = Path("data/processed/Test_Data.csv")
METADATA_PATH = Path("data/processed/classification_split_metadata.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_encoder() -> OneHotEncoder:
    """Support the sklearn versions used by group members."""
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - only for older sklearn versions
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def prepare_classification_splits(
    raw_path: str | Path = RAW_PATH,
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
    metadata_path: str | Path = METADATA_PATH,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split raw Telco data then fit preprocessing exclusively on Train.

    Returns the exported train/test frames and metadata used for auditability.
    """
    raw_path = Path(raw_path)
    train_path = Path(train_path)
    test_path = Path(test_path)
    metadata_path = Path(metadata_path)
    raw = pd.read_csv(raw_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce")
    raw["Churn"] = raw["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if raw["Churn"].isna().any():
        raise ValueError("Churn must contain only Yes/No values.")

    identifiers = raw["customerID"].copy()
    target = raw["Churn"].astype(int)
    features = raw.drop(columns=["customerID", "Churn"])
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        features,
        target,
        identifiers,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(numeric_imputer.fit_transform(X_train[numeric_columns]))
    X_test_numeric = scaler.transform(numeric_imputer.transform(X_test[numeric_columns]))

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    encoder = _make_encoder()
    X_train_categorical = encoder.fit_transform(
        categorical_imputer.fit_transform(X_train[categorical_columns])
    )
    X_test_categorical = encoder.transform(
        categorical_imputer.transform(X_test[categorical_columns])
    )

    encoded_columns = encoder.get_feature_names_out(categorical_columns).tolist()
    train_features = pd.DataFrame(
        X_train_numeric, columns=numeric_columns, index=X_train.index
    ).join(pd.DataFrame(X_train_categorical, columns=encoded_columns, index=X_train.index))
    test_features = pd.DataFrame(
        X_test_numeric, columns=numeric_columns, index=X_test.index
    ).join(pd.DataFrame(X_test_categorical, columns=encoded_columns, index=X_test.index))

    train_output = train_features.copy()
    train_output.insert(0, "customerID", id_train)
    train_output["Churn"] = y_train
    test_output = test_features.copy()
    test_output.insert(0, "customerID", id_test)
    test_output["Churn"] = y_test

    if set(train_output["customerID"]).intersection(test_output["customerID"]):
        raise AssertionError("Train and test customerID values must be disjoint.")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)
    metadata: dict[str, object] = {
        "source_file": str(raw_path).replace("\\", "/"),
        "source_sha256": _file_sha256(raw_path),
        "source_rows": int(len(raw)),
        "train_rows": int(len(train_output)),
        "test_rows": int(len(test_output)),
        "test_size": test_size,
        "random_state": random_state,
        "stratify": "Churn",
        "numeric_imputer_fit_on": "Train_Data only",
        "scaler_fit_on": "Train_Data only",
        "categorical_imputer_fit_on": "Train_Data only",
        "encoder_fit_on": "Train_Data only",
        "numeric_columns": numeric_columns,
        "encoded_feature_count": len(encoded_columns),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_output, test_output, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare leakage-safe Telco classification splits.")
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--train", default=str(TRAIN_PATH))
    parser.add_argument("--test", default=str(TEST_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    args = parser.parse_args()
    train_frame, test_frame, audit = prepare_classification_splits(
        args.raw, args.train, args.test, args.metadata
    )
    print(f"Created train/test: {len(train_frame)} / {len(test_frame)} rows")
    print(f"random_state={audit['random_state']}, stratify={audit['stratify']}")
