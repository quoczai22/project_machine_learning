import pandas as pd
import pytest

from src.association_data import build_transaction_table


def _telco_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "tenure": [2, 60],
            "MonthlyCharges": [25.0, 89.0],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["Yes", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "Churn": ["Yes", "No"],
        }
    )


def test_transaction_table_excludes_customer_id_and_keeps_churn():
    transactions = build_transaction_table(_telco_rows())

    assert transactions.shape[0] == 2
    assert transactions.dtypes.eq(bool).all()
    assert not any("customerID" in column for column in transactions.columns)
    assert "Churn=Yes" in transactions.columns
    assert "Churn=No" in transactions.columns
    assert "tenure_band=0-12 tháng" in transactions.columns
    assert "monthly_charge_band=cao (>70)" in transactions.columns


def test_transaction_table_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Thiếu cột"):
        build_transaction_table(_telco_rows().drop(columns="Contract"))
