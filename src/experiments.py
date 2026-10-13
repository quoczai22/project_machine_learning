"""Classification experiment using member 2's paired train/test datasets.

Thực thi thực nghiệm phân lớp với Audit Logging toàn diện.
Xuất bảng đánh giá độ đo (Accuracy, Precision, Recall, F1-Score) ra CSV và biểu đồ Confusion Matrix.
The train and test CSVs have already been split and preprocessed. This module
must therefore not split, encode, scale, or impute them a second time.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42
TRAIN_PATH = "data/processed/Train_Data.csv"
TEST_PATH = "data/processed/Test_Data.csv"


def setup_logger(log_file: str = "logs/experiments_audit.log") -> logging.Logger:
    """Configure an audit logger for the experiment that writes to both console and file."""
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
    figures_dir: str = "reports/figures",
) -> pd.DataFrame:
    """Tune on Train_Data and report final metrics on the paired Test_Data."""
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU THỰC NGHIỆM PHÂN LỚP (CLASSIFICATION EXPERIMENT)")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_preprocessed_splits(train_path, test_path)
    logger.info("Đã tải dữ liệu | Train: %s dòng | Test: %s dòng", len(X_train), len(X_test))

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

    results_list: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    for name, (model, parameter_grid) in models.items():
        logger.info(f"Đang tối ưu mô hình: {name} (GridSearchCV)...")
        start_time = time.time()
        search = GridSearchCV(model, parameter_grid, cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        duration = time.time() - start_time

        best_estimator = search.best_estimator_
        fitted_models[name] = best_estimator
        prediction = best_estimator.predict(X_test)

        acc = accuracy_score(y_test, prediction)
        prec = precision_score(y_test, prediction)
        rec = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)

        logger.info(
            f"Mô hình: {name.upper()} | Time: {duration:.2f}s | Params: {search.best_params_} | "
            f"CV_F1: {search.best_score_:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        )

        results_list.append({
            "Model": name,
            "Best_Params": str(search.best_params_),
            "CV_F1_Score": round(search.best_score_, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
        })

    df_results = pd.DataFrame(results_list)

    # Lưu bảng kết quả metrics
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.info(f"[LƯU THÀNH CÔNG] Bảng kết quả metrics lưu tại: {output_path}")

    # Vẽ và lưu Confusion Matrix
    os.makedirs(figures_dir, exist_ok=True)
    for name, model in fitted_models.items():
        prediction = model.predict(X_test)
        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, prediction), display_labels=["No Churn", "Churn"]
        ).plot(ax=axis, cmap="Blues" if "Logistic" in name else "Greens", values_format="d")
        axis.set_title(f"Confusion Matrix: {name}")
        figure.tight_layout()
        cm_path = os.path.join(figures_dir, f"cm_{name.lower().replace(' ', '_')}.png")
        figure.savefig(cm_path)
        plt.close(figure)
        logger.info(f"[LƯU THÀNH CÔNG] Confusion matrix lưu tại: {cm_path}")

    logger.info("=" * 60)
    return df_results


if __name__ == "__main__":
    run_classification_experiments()
