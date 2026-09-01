"""Classification experiment using member 2's paired train/test datasets.

<<<<<<< HEAD
Thực thi thực nghiệm phân lớp với Audit Logging toàn diện.
Xuất bảng đánh giá độ đo (Accuracy, Precision, Recall, F1-Score) ra CSV.
"""

import logging
import os
import sys
import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.classification import predict_classifiers, tune_classifiers


def setup_logger(log_file: str = "logs/experiments_audit.log") -> logging.Logger:
    """Cấu hình hệ thống Audit Log ghi ra cả Terminal lẫn File Log.

    Args:
        log_file (str): Đường dẫn lưu file log.

    Returns:
        logging.Logger: Object logger đã được cấu hình.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("ClassificationAudit")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream Handler (Hiển thị Console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Lưu file log)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def run_classification_experiments(
    output_path: str = "data/processed/classification_results.csv",
) -> pd.DataFrame:
    """Chạy luồng thực nghiệm phân lớp có Audit Log đầy đủ."""
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("KHI BẮT ĐẦU THỰC NGHIỆM PHÂN LỚP (CLASSIFICATION EXPERIMENT)")
    logger.info("=" * 60)

    # 1. Đọc dữ liệu
    data_path = "data/processed/telco_train_data.csv"
    if not os.path.exists(data_path):
        data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    logger.info(f"[1/5] Đang đọc dữ liệu từ đường dẫn: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"-> Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")

    # 2. Xử lý chuẩn hóa và kiểm tra dữ liệu an toàn
    logger.info("[2/5] Tiến hành ép kiểu và làm sạch dữ liệu khuyết...")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.dropna(subset=["Churn"])

    if df["Churn"].dtype == "object":
        df["Churn"] = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        )
    df = df.dropna(subset=["Churn"])

    df_clean = df.drop(columns=["customerID"], errors="ignore")
    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"].astype(int)

    # 3. Phân chia Train/Test
    logger.info("[3/5] Phân chia dữ liệu Train/Test (Tỷ lệ 80/20, Stratify)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"-> Tập Train: {X_train.shape[0]} mẫu | Tập Test: {X_test.shape[0]} mẫu")

    # 4. Huấn luyện và Tối ưu tham số (GridSearchCV)
    logger.info("[4/5] Khởi chạy Cross-Validation & GridSearch tối ưu cho mô hình...")
    start_train_time = time.time()
    
    tuned_results = tune_classifiers(X_train, y_train, cv=5)
    
    train_duration = time.time() - start_train_time
    logger.info(f"-> Hoàn tất quá trình GridSearch trong: {train_duration:.2f} giây")

    # 5. Thực hiện dự đoán trên Test Set
    logger.info("[5/5] Dự đoán và tính toán các chỉ số đánh giá (Metrics)...")
    predictions = predict_classifiers(tuned_results, X_test)

    results_list = []
    for model_name, model_info in tuned_results.items():
        y_pred = predictions[model_name]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(
            f"Mô hình: {model_name.upper()} | Params: {model_info['best_params']} | "
            f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
        )

        results_list.append(
            {
                "Model": model_name,
                "Best_Params": str(model_info["best_params"]),
                "CV_F1_Score": round(model_info["cv_score"], 4),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
            }
        )

    df_results = pd.DataFrame(results_list)

    # Lưu bảng metrics
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.info(f"[LƯU THÀNH CÔNG] Bảng kết quả metrics lưu tại: {output_path}")
    logger.info("=" * 60)

    return df_results
=======
The train and test CSVs have already been split and preprocessed. This module
must therefore not split, encode, scale, or impute them a second time.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


RANDOM_STATE = 42
TRAIN_PATH = "data/processed/Train_Data.csv"
TEST_PATH = "data/processed/Test_Data.csv"


def setup_logger(log_file: str = "logs/experiments_audit.log") -> logging.Logger:
    """Configure an audit logger for the experiment."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("ClassificationAudit")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_preprocessed_splits(
    train_path: str = TRAIN_PATH, test_path: str = TEST_PATH
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the paired 80/20 splits without changing their preprocessing."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    required_columns = {"customerID", "Churn"}
    if not required_columns.issubset(train_data.columns) or not required_columns.issubset(test_data.columns):
        raise ValueError("Each split must contain customerID and Churn.")

    X_train = train_data.drop(columns=["customerID", "Churn"])
    X_test = test_data.drop(columns=["customerID", "Churn"])
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns must match exactly.")
    return X_train, X_test, train_data["Churn"].astype(int), test_data["Churn"].astype(int)


def run_classification_experiments(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    output_path: str = "data/processed/classification_results.csv",
) -> pd.DataFrame:
    """Tune on Train_Data and report final metrics on the paired Test_Data."""
    logger = setup_logger()
    X_train, X_test, y_train, y_test = load_preprocessed_splits(train_path, test_path)
    logger.info("Train: %s rows | Test: %s rows", len(X_train), len(X_test))

    models = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]},
        ),
        "Random Forest Classifier": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15, None], "min_samples_split": [2, 5]},
        ),
    }

    results: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}
    for name, (model, parameter_grid) in models.items():
        search = GridSearchCV(model, parameter_grid, cv=5, scoring="f1", n_jobs=1)
        search.fit(X_train, y_train)
        prediction = search.best_estimator_.predict(X_test)
        fitted_models[name] = search.best_estimator_
        results.append({
            "Model": name,
            "Best_Params": str(search.best_params_),
            "CV_F1_Score": round(search.best_score_, 4),
            "Accuracy": round(accuracy_score(y_test, prediction), 4),
            "Precision": round(precision_score(y_test, prediction), 4),
            "Recall": round(recall_score(y_test, prediction), 4),
            "F1_Score": round(f1_score(y_test, prediction), 4),
        })

    result_frame = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_frame.to_csv(output_path, index=False)
    os.makedirs("reports/figures", exist_ok=True)
    for name, model in fitted_models.items():
        prediction = model.predict(X_test)
        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, prediction), display_labels=["No Churn", "Churn"]
        ).plot(ax=axis, cmap="Blues" if "Logistic" in name else "Greens", values_format="d")
        axis.set_title(f"Confusion Matrix: {name}")
        figure.tight_layout()
        figure.savefig(f"reports/figures/cm_{name.lower().replace(' ', '_')}.png")
        plt.close(figure)
    return result_frame
>>>>>>> branch_bao


if __name__ == "__main__":
    run_classification_experiments()
