"""Loading and validation utilities for Telco Churn data."""

from pathlib import Path

import pandas as pd


REQUIRED_RAW_COLUMNS = {"customerID", "TotalCharges", "Churn"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a new DataFrame.

    Args:
        path: Location of the source CSV file.

    Returns:
        The data in the CSV file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If essential Telco columns are absent.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    missing_columns = REQUIRED_RAW_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return data
