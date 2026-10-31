"""Tạo lại model và bộ tiền xử lý cho demo web.

Quy trình giống hệt project chính (telco-merge-elbow):
- Phân lớp: split 80/20, random_state=42, stratify=Churn; imputer/scaler/encoder
  chỉ fit trên Train; Logistic Regression và Random Forest với tham số tốt nhất
  mà GridSearchCV đã chọn (classification_results.csv).
- Gom cụm: tenure, MonthlyCharges, service_count; StandardScaler + K-Means k=4,
  random_state=42, n_init=20.

Sau khi train, script so khớp số liệu với kết quả đã chốt và dừng nếu lệch.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
RAW_PATH = BASE.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ART = BASE / "artifacts"
RANDOM_STATE = 42

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "service_count"]

# Số liệu đã chốt trong reports/metrics của project chính.
EXPECTED_CLASSIFICATION = {
    "Logistic Regression": {"Accuracy": 0.7424, "Precision": 0.5096, "Recall": 0.7834, "F1": 0.6175},
    "Random Forest": {"Accuracy": 0.7530, "Precision": 0.5236, "Recall": 0.7701, "F1": 0.6234},
}
EXPECTED_SILHOUETTE_K4 = 0.4212
EXPECTED_CLUSTER_SIZES = {0: 1557, 1: 2144, 2: 2315, 3: 1027}

# Tên phân khúc theo slide "Bốn phân khúc khách hàng (K-Means, k = 4)".
SEGMENT_NAMES = {
    0: "Khách mới, phí thấp",
    1: "Gắn bó lâu, phí cao, nhiều dịch vụ",
    2: "Khách mới, phí khá cao",
    3: "Gắn bó lâu, phí thấp",
}


def load_raw() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce")
    raw["Churn"] = raw["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1}).astype(int)
    return raw


def service_count(frame: pd.DataFrame) -> pd.Series:
    """Số dịch vụ khách đang dùng = số cột dịch vụ có giá trị 'Yes'."""
    return (frame[SERVICE_COLUMNS] == "Yes").sum(axis=1)


class TelcoPreprocessor:
    """Tiền xử lý phân lớp, fit trên Train giống src/prepare_classification_data.py."""

    def fit(self, X: pd.DataFrame) -> "TelcoPreprocessor":
        self.numeric_columns = X.select_dtypes(include="number").columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
        self.numeric_imputer = SimpleImputer(strategy="median").fit(X[self.numeric_columns])
        self.scaler = StandardScaler().fit(self.numeric_imputer.transform(X[self.numeric_columns]))
        self.categorical_imputer = SimpleImputer(strategy="most_frequent").fit(X[self.categorical_columns])
        self.encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False).fit(
            self.categorical_imputer.transform(X[self.categorical_columns])
        )
        self.feature_columns = self.numeric_columns + self.encoder.get_feature_names_out(
            self.categorical_columns
        ).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = self.scaler.transform(self.numeric_imputer.transform(X[self.numeric_columns]))
        categorical = self.encoder.transform(self.categorical_imputer.transform(X[self.categorical_columns]))
        return pd.DataFrame(np.hstack([numeric, categorical]), columns=self.feature_columns, index=X.index)


def _scores(y_true, y_pred) -> dict[str, float]:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
    }


def build() -> dict:
    ART.mkdir(exist_ok=True)
    raw = load_raw()

    # ---------- Phân lớp ----------
    features = raw.drop(columns=["customerID", "Churn"])
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        features, raw["Churn"], raw["customerID"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=raw["Churn"],
    )
    prep = TelcoPreprocessor().fit(X_train)
    Xt_train, Xt_test = prep.transform(X_train), prep.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.1, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=2,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }
    classification = {}
    for name, model in models.items():
        model.fit(Xt_train, y_train)
        pred = model.predict(Xt_test)
        scores = _scores(y_test, pred)
        expected = EXPECTED_CLASSIFICATION[name]
        if any(abs(scores[k] - expected[k]) > 1e-4 for k in expected):
            raise AssertionError(f"{name}: số liệu lệch kết quả đã chốt: {scores} != {expected}")
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        classification[name] = {**scores, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

    test_samples = X_test.copy()
    test_samples.insert(0, "customerID", id_test)
    test_samples["Churn"] = y_test

    # ---------- Gom cụm ----------
    cluster_data = raw[["tenure", "MonthlyCharges"]].copy()
    cluster_data["service_count"] = service_count(raw)
    cluster_scaler = StandardScaler().fit(cluster_data[CLUSTER_FEATURES])
    X_cluster = cluster_scaler.transform(cluster_data[CLUSTER_FEATURES])
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20).fit(X_cluster)
    labels = kmeans.labels_
    silhouette = round(float(silhouette_score(X_cluster, labels)), 4)
    sizes = {int(k): int(v) for k, v in pd.Series(labels).value_counts().items()}
    if abs(silhouette - EXPECTED_SILHOUETTE_K4) > 1e-4 or sizes != EXPECTED_CLUSTER_SIZES:
        raise AssertionError(f"K-Means lệch kết quả đã chốt: silhouette={silhouette}, sizes={sizes}")

    cluster_points = cluster_data.assign(
        cluster=labels, Churn=raw["Churn"].values, customerID=raw["customerID"].values
    )
    profile = (
        cluster_points.groupby("cluster")
        .agg(
            count=("tenure", "size"),
            tenure_mean=("tenure", "mean"),
            MonthlyCharges_mean=("MonthlyCharges", "mean"),
            service_count_mean=("service_count", "mean"),
            churn_rate=("Churn", "mean"),
        )
        .reset_index()
    )
    profile["segment"] = profile["cluster"].map(SEGMENT_NAMES)

    # ---------- Lưu ----------
    joblib.dump(prep, ART / "preprocessor.joblib")
    joblib.dump(models, ART / "classifiers.joblib")
    joblib.dump({"scaler": cluster_scaler, "kmeans": kmeans}, ART / "clustering.joblib")
    test_samples.to_csv(ART / "test_samples.csv", index=False)
    cluster_points.to_csv(ART / "cluster_points.csv", index=False)
    profile.to_csv(ART / "cluster_profile.csv", index=False)
    summary = {
        "rows": int(len(raw)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "churn_rate": round(float(raw["Churn"].mean()), 4),
        "classification": classification,
        "silhouette_k4": silhouette,
        "segment_names": {str(k): v for k, v in SEGMENT_NAMES.items()},
    }
    (ART / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    # Import lại chính module để class được pickle với tên "build_artifacts",
    # nhờ vậy ứng dụng web nạp được preprocessor.joblib.
    import build_artifacts

    result = build_artifacts.build()
    print("Đã tạo artifacts, số liệu khớp kết quả đã chốt:")
    for name, scores in result["classification"].items():
        print(f"  {name}: {scores}")
    print(f"  K-Means k=4 Silhouette: {result['silhouette_k4']}")
