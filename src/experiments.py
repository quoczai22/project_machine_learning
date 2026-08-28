"""Module experiments.py.

Chạy thực nghiệm so sánh hai mô hình Logistic Regression và Random Forest
Sử dụng GridSearchCV để tìm bộ tham số tối ưu nhất, tính toán điểm F1-Score,
thời gian chạy và dự đoán trên tập kiểm thử (Test Set).
"""

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessor import build_classification_preprocessor


def run_classification_experiments():
    print("=" * 60)
    print("BẮT ĐẦU TIẾN TRÌNH THỰC NGHIỆM PHÂN LỚP (CLASSIFICATION EXPERIMENTS)")
    print("=" * 60)

    # 1. ĐỌC DỮ LIỆU
    data_path = "data/processed/telco_train_data.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Dự phòng đường dẫn nếu chạy từ thư mục gốc dự án
        data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = pd.read_csv(data_path)

    # 2. TIỀN XỬ LÝ AN TOÀN TRÁNH LỖI KHI TÁCH TRAIN/TEST
    # Ép kiểu sang string trước khi strip để tránh lỗi AttributeError đối với cột kiểu số
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Loại bỏ dòng bị trống ở cột Churn (nếu có)
    df = df.dropna(subset=["Churn"])

    # Map nhãn Churn sang dạng số (1/0)
    if df["Churn"].dtype == "object":
        df["Churn"] = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        )

    # Loại bỏ các dòng map thất bại gây ra NaN ở y
    df = df.dropna(subset=["Churn"])

    # Tách X và y
    df_clean = df.drop(columns=["customerID"], errors="ignore")
    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"].astype(int)

    # 3. PHÂN CHIA TẬP TRAIN / TEST (random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"-> Tập Train: {X_train.shape[0]} mẫu")
    print(f"-> Tập Test:  {X_test.shape[0]} mẫu\n")

    # 4. XÂY DỰNG PIPELINE PREPROCESSOR
    preprocessor = build_classification_preprocessor(X_train)

    # ---------------------------------------------------------
    # MÔ HÌNH 1: LOGISTIC REGRESSION
    # ---------------------------------------------------------
    print("--- [1/2] Huấn luyện & Tối ưu Logistic Regression ---")
    pipe_lr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", random_state=42, max_iter=1000
                ),
            ),
        ]
    )

    param_grid_lr = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    start_time_lr = time.time()
    grid_lr = GridSearchCV(
        estimator=pipe_lr,
        param_grid=param_grid_lr,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid_lr.fit(X_train, y_train)
    lr_fit_time = time.time() - start_time_lr

    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)

    # ---------------------------------------------------------
    # MÔ HÌNH 2: RANDOM FOREST CLASSIFIER
    # ---------------------------------------------------------
    print("--- [2/2] Huấn luyện & Tối ưu Random Forest Classifier ---")
    pipe_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    param_grid_rf = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5],
    }

    start_time_rf = time.time()
    grid_rf = GridSearchCV(
        estimator=pipe_rf,
        param_grid=param_grid_rf,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    rf_fit_time = time.time() - start_time_rf

    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    # ---------------------------------------------------------
    # XUẤT TỔNG KẾT KẾT QUẢ THỰC NGHIỆM
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("TỔNG HỢP KẾT QUẢ THỰC NGHIỆM")
    print("=" * 60)

    results = [
        {
            "Model": "Logistic Regression",
            "Best Params": grid_lr.best_params_,
            "CV F1-Score": round(grid_lr.best_score_, 4),
            "Time (s)": round(lr_fit_time, 2),
            "Test Accuracy": round(accuracy_score(y_test, y_pred_lr), 4),
            "Test Precision": round(precision_score(y_test, y_pred_lr), 4),
            "Test Recall": round(recall_score(y_test, y_pred_lr), 4),
            "Test F1-Score": round(f1_score(y_test, y_pred_lr), 4),
        },
        {
            "Model": "Random Forest Classifier",
            "Best Params": grid_rf.best_params_,
            "CV F1-Score": round(grid_rf.best_score_, 4),
            "Time (s)": round(rf_fit_time, 2),
            "Test Accuracy": round(accuracy_score(y_test, y_pred_rf), 4),
            "Test Precision": round(precision_score(y_test, y_pred_rf), 4),
            "Test Recall": round(recall_score(y_test, y_pred_rf), 4),
            "Test F1-Score": round(f1_score(y_test, y_pred_rf), 4),
        },
    ]

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run_classification_experiments()