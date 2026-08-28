"""Module experiments.py.

Thực thi thực nghiệm phân lớp bằng cách tái sử dụng hàm từ classification.py.
Trả về và lưu kết quả đánh giá (CSV) để người tiếp theo (người 4/5) sử dụng.
"""

import os
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


def run_classification_experiments(
    output_path: str = "data/processed/classification_results.csv",
) -> pd.DataFrame:
    """Chạy toàn bộ luồng thực nghiệm phân lớp, trả về và lưu bảng kết quả.

    Args:
        output_path (str): Đường dẫn file CSV lưu kết quả.

    Returns:
        pd.DataFrame: Bảng tổng hợp kết quả chi tiết của các mô hình.
    """
    print("=" * 60)
    print("BẮT ĐẦU TIẾN TRÌNH THỰC NGHIỆM PHÂN LỚP")
    print("=" * 60)

    # 1. Đọc dữ liệu
    data_path = "data/processed/telco_train_data.csv"
    if not os.path.exists(data_path):
        data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = pd.read_csv(data_path)

    # 2. Xử lý an toàn các cột
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

    # 3. Phân chia Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Gọi logic huấn luyện & tối ưu từ classification.py
    start_time = time.time()
    tuned_results = tune_classifiers(X_train, y_train, cv=5)
    execution_time = time.time() - start_time

    # Dự đoán nhãn
    predictions = predict_classifiers(tuned_results, X_test)

    # 5. Tổng hợp kết quả
    results_list = []
    for model_name, model_info in tuned_results.items():
        y_pred = predictions[model_name]

        results_list.append(
            {
                "Model": model_name,
                "Best_Params": str(model_info["best_params"]),
                "CV_F1_Score": round(model_info["cv_score"], 4),
                "Test_Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Test_Precision": round(precision_score(y_test, y_pred), 4),
                "Test_Recall": round(recall_score(y_test, y_pred), 4),
                "Test_F1_Score": round(f1_score(y_test, y_pred), 4),
            }
        )

    df_results = pd.DataFrame(results_list)

    # 6. Lưu file kết quả cho người 4/5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n[OK] Đã lưu bảng kết quả thực nghiệm tại: {output_path}")

    print("\n" + "=" * 60)
    print("TỔNG HỢP KẾT QUẢ THỰC NGHIỆM")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)

    return df_results


if __name__ == "__main__":
    run_classification_experiments()