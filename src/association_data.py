"""Chuẩn bị dữ liệu Telco dưới dạng giao dịch cho luật kết hợp.

Mỗi hàng là một khách hàng; mỗi cột Boolean biểu diễn một mục dạng
``ten_cot=gia_tri``. Module này chỉ chuẩn bị dữ liệu, không sinh luật.
"""
from __future__ import annotations

import pandas as pd


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
CONTEXT_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn",
]
REQUIRED_COLUMNS = {"customerID", "tenure", "MonthlyCharges", *SERVICE_COLUMNS, *CONTEXT_COLUMNS}


def _tenure_band(values: pd.Series) -> pd.Series:
    """Chia thâm niên theo các khoảng dễ diễn giải trong báo cáo."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("tenure có giá trị thiếu hoặc không phải số")
    return pd.cut(
        numeric,
        bins=[-1, 12, 48, float("inf")],
        labels=["0-12 tháng", "13-48 tháng", ">48 tháng"],
    ).astype("string")


def _monthly_charge_band(values: pd.Series) -> pd.Series:
    """Chia phí tháng thành thấp, trung bình và cao bằng ngưỡng cố định."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("MonthlyCharges có giá trị thiếu hoặc không phải số")
    return pd.cut(
        numeric,
        bins=[-float("inf"), 35, 70, float("inf")],
        labels=["thấp (<=35)", "trung bình (35-70)", "cao (>70)"],
        include_lowest=True,
    ).astype("string")


def build_transaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Trả về ma trận giao dịch Boolean từ dữ liệu Telco thô/đã làm sạch.

    ``customerID`` bị loại hoàn toàn. ``tenure`` và ``MonthlyCharges`` được
    chia khoảng trước khi one-hot encode để phù hợp với Apriori/FP-Growth.
    """
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột cho luật kết hợp: {sorted(missing)}")

    items = df[SERVICE_COLUMNS + CONTEXT_COLUMNS].copy()
    items["SeniorCitizen"] = items["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(
        items["SeniorCitizen"].astype("string")
    )
    items["tenure_band"] = _tenure_band(df["tenure"])
    items["monthly_charge_band"] = _monthly_charge_band(df["MonthlyCharges"])

    if items.isna().any().any():
        raise ValueError("Dữ liệu luật kết hợp còn thiếu giá trị")
    encoded = pd.get_dummies(items.astype("string"), prefix=items.columns, prefix_sep="=")
    return encoded.astype(bool)
