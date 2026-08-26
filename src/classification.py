"""
Module classification.py
-------------------------
Xây dựng, tối ưu hóa (GridSearchCV) và dự đoán các mô hình phân lớp 
(Logistic Regression và Random Forest) cho bài toán Telco Customer Churn.
"""

from typing import Any, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessor import build_classification_preprocessor


def build_classification_pipelines(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """
    Tạo dictionary chứa các Pipeline chưa fit cho Logistic Regression và Random Forest.
    Cả hai mô hình đều bắt buộc thiết lập class_weight="balanced".

    Args:
        X (pd.DataFrame): Tập đặc trưng đầu vào (chưa qua transform) để xây dựng preprocessor.

    Returns:
        Dict[str, Pipeline]: Dictionary chứa các Pipeline dạng:
            {
                "logistic_regression": Pipeline(...),
                "random_forest": Pipeline(...)
            }
    """
    preprocessor = build_classification_preprocessor(X)

    # 1. Pipeline Logistic Regression
    lr_pipeline = Pipeline(
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

    # 2. Pipeline Random Forest
    rf_pipeline = Pipeline(
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

    return {
        "logistic_regression": lr_pipeline,
        "random_forest": rf_pipeline,
    }


def tune_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Thực hiện GridSearchCV trên tập X_train, y_train để tối ưu tham số cho từng model.
    Không nhìn thấy tập test (X_test / y_test) để tránh Data Leakage.

    Args:
        X_train (pd.DataFrame): Tập đặc trưng huấn luyện.
        y_train (pd.Series): Nhãn mục tiêu huấn luyện (0 và 1).
        cv (int, optional): Số fold Cross-Validation. Mặc định là 5.
        random_state (int, optional): Random seed phục vụ tái tạo kết quả. Mặc định là 42.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary lưu trữ kết quả tinh chỉnh cho từng model:
            {
                "logistic_regression": {
                    "estimator": Pipeline (đã fit tốt nhất),
                    "best_params": dict,
                    "cv_score": float (F1-score trung bình trên CV)
                },
                "random_forest": {
                    "estimator": Pipeline (đã fit tốt nhất),
                    "best_params": dict,
                    "cv_score": float (F1-score trung bình trên CV)
                }
            }
    """
    pipelines = build_classification_pipelines(X_train)

    # Lưới tham số tìm kiếm cho Logistic Regression
    param_grid_lr = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    # Lưới tham số tìm kiếm cho Random Forest
    param_grid_rf = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5],
    }

    param_grids = {
        "logistic_regression": param_grid_lr,
        "random_forest": param_grid_rf,
    }

    tuned_results: Dict[str, Dict[str, Any]] = {}

    for model_name, pipeline in pipelines.items():
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="f1",  # Tối ưu hóa F1-score do dữ liệu mất cân bằng
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        tuned_results[model_name] = {
            "estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "cv_score": float(grid_search.best_score_),
        }

    return tuned_results


def predict_classifiers(
    fitted_models: Dict[str, Any], X_test: pd.DataFrame
) -> Dict[str, pd.Series]:
    """
    Tạo nhãn dự đoán 0/1 trên tập X_test bằng các mô hình đã fit tốt nhất.

    Args:
        fitted_models (Dict[str, Any]): Dictionary chứa kết quả từ `tune_classifiers` 
                                       hoặc trực tiếp chứa các pipeline đã fit.
        X_test (pd.DataFrame): Tập đặc trưng kiểm thử.

    Returns:
        Dict[str, pd.Series]: Dictionary chứa nhãn dự đoán dạng pd.Series giữ nguyên index của X_test:
            {
                "logistic_regression": pd.Series([0, 1, 0, ...], index=X_test.index),
                "random_forest": pd.Series([0, 1, 1, ...], index=X_test.index)
            }
    """
    predictions: Dict[str, pd.Series] = {}

    for model_name, model_info in fitted_models.items():
        # Xử lý trường hợp đầu vào là dict trả về từ tune_classifiers hoặc trực tiếp là Pipeline
        if isinstance(model_info, dict) and "estimator" in model_info:
            estimator = model_info["estimator"]
        else:
            estimator = model_info

        y_pred = estimator.predict(X_test)
        predictions[model_name] = pd.Series(y_pred, index=X_test.index, name=model_name)

    return predictions