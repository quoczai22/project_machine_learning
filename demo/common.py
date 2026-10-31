"""Hàm dùng chung cho các trang demo: nạp artifacts và cấu hình trang."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import build_artifacts
from build_artifacts import SERVICE_COLUMNS, TelcoPreprocessor  # noqa: F401  (cần cho joblib)

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
REQUIRED = [
    "preprocessor.joblib", "classifiers.joblib", "clustering.joblib",
    "test_samples.csv", "cluster_points.csv", "cluster_profile.csv", "summary.json",
]


def setup_page(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} · Telco Churn", page_icon=icon, layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_artifacts() -> dict:
    if not all((ART / name).exists() for name in REQUIRED):
        build_artifacts.build()
    return {
        "preprocessor": joblib.load(ART / "preprocessor.joblib"),
        "classifiers": joblib.load(ART / "classifiers.joblib"),
        "clustering": joblib.load(ART / "clustering.joblib"),
        "test_samples": pd.read_csv(ART / "test_samples.csv"),
        "cluster_points": pd.read_csv(ART / "cluster_points.csv"),
        "cluster_profile": pd.read_csv(ART / "cluster_profile.csv"),
        "summary": json.loads((ART / "summary.json").read_text(encoding="utf-8")),
    }


def fmt_pct(value: float, digits: int = 1) -> str:
    """Định dạng phần trăm kiểu Việt Nam: 26,5%."""
    return f"{value * 100:.{digits}f}%".replace(".", ",")
