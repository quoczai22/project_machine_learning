"""Module experiments.py.

Đọc dữ liệu thô 7,043 dòng, phân chia Train/Test trước khi biến đổi,
sử dụng Pipeline & GridSearchCV để chống Data Leakage và lưu kết quả.
"""

import logging
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def setup_logger(log_file: str = "logs/experiments_audit.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("ClassificationAudit")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def run_classification_experiments(
    data_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_path: str = "data/processed/classification_results.csv",
) -> pd.DataFrame:
    """Hàm chạy thực nghiệm chính xuất ra DataFrame kết quả."""
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("KHI BẮT ĐẦU THỰC NGHIỆM PHÂN LỚP KHÔNG RÒ RỈ DỮ LIỆU (NO LEAKAGE)")
    logger.info("=" * 60)

    # Đọc dữ liệu gốc
    if not os.path.exists(data_path):
        data_path = "data/processed/telco_train_data.csv"

    df_raw = pd.read_csv(data_path)
    logger.info(f"Đã đọc dữ liệu thành công: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột.")

    # Tiền xử lý dữ liệu thô
    df_raw["TotalCharges"] = pd.to_numeric(
        df_raw["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df_raw = df_raw.dropna(subset=["Churn"])

    X = df_raw.drop(columns=["customerID", "Churn"], errors="ignore")
    y = (
        df_raw["Churn"]
        .astype(str)
        .str.strip()
        .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        .astype(int)
    )

    # Phân chia Train / Test trước khi biến đổi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Định nghĩa ColumnTransformer trong Pipeline
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    # Khởi tạo mô hình
    pipe_lr = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    param_grid_lr = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])
    param_grid_rf = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5],
    }

    models = {
        "Logistic Regression": (pipe_lr, param_grid_lr),
        "Random Forest Classifier": (pipe_rf, param_grid_rf),
    }

    results = []
    best_estimators = {}

    for name, (pipe, param_grid) in models.items():
        logger.info(f"Đang huấn luyện mô hình: {name}...")
        grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_estimators[name] = best_model
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Best_Params": str(grid.best_params_),
            "CV_F1_Score": round(grid.best_score_, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
        })

    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.info(f"Đã lưu kết quả tại: {output_path}")

    # Xuất ma trận nhầm lẫn
    os.makedirs("reports/figures", exist_ok=True)
    for name, model in best_estimators.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, cmap="Blues" if "Logistic" in name else "Greens", values_format="d")
        ax.set_title(f"Confusion Matrix: {name}")
        file_name = f"reports/figures/cm_{name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    return df_results


if __name__ == "__main__":
    run_classification_experiments()