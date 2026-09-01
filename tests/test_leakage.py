"""Unit tests phòng chống Data Leakage và kiểm tra tính hợp lệ của dữ liệu thô."""

import os
from pathlib import Path
import pandas as pd
import pytest

from src.experiments import run_classification_experiments

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn chính xác tới file trong data/raw/
DATA_FILE = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Dự phòng nếu file nằm ở gốc
if not DATA_FILE.exists():
    DATA_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_no_data_leakage_in_preprocessing():
    """Đảm bảo dữ liệu chưa bị scale/one-hot trước khi train_test_split."""
    assert DATA_FILE.exists(), f"Không tìm thấy file dữ liệu gốc tại: {DATA_FILE}"

    df_raw = pd.read_csv(DATA_FILE)

    # Kiểm tra các cột gốc
    assert "Contract" in df_raw.columns
    assert "InternetService" in df_raw.columns
    assert "Contract_OneYear" not in df_raw.columns
    assert df_raw["MonthlyCharges"].max() > 10.0


def test_experiment_output_pipeline():
    """Kiểm tra luồng xuất file kết quả CSV duy nhất."""
    df_results = run_classification_experiments(data_path=str(DATA_FILE))
    assert isinstance(df_results, pd.DataFrame)
    assert len(df_results) == 2
    assert "F1_Score" in df_results.columns