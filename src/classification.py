"""
Module classification.py
-------------------------
Xây dựng, tối ưu hóa siêu tham số (GridSearchCV) và dự đoán nhãn phân lớp 
sử dụng Hồi quy Logic (Logistic Regression) và Hồi quy Tuyến tính (Linear Regression).
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessor import build_classification_preprocessor


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper chuyển đổi mô hình Hồi quy Tuyến tính (Linear Regression) 
    thành Mô hình Phân lớp Nhị phân dựa trên ngưỡng xác suất (Threshold = 0.5).
    """

    def __init__(self, threshold: float = 0.5, fit_intercept: bool = True):
        self.threshold = threshold
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=self.fit_intercept)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_preds = self.model.predict(X)
        return (raw_preds >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_preds = self.model.predict(X)
        # Clip giá trị nằm trong khoảng [0, 1] để giả định làm xác suất
        probs_class_1 = np.clip(raw_preds, 0, 1)
        probs_class_0 = 1.0 - probs_class_1
        return np.column_stack((probs_class_0, probs_class_1))


def build_classification_pipelines(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """
    Tạo dictionary chứa các Pipeline chưa fit cho Logistic Regression và Linear Regression Classifier.

    Args:
        X (pd.DataFrame): Tập đặc trưng đầu vào để xây dựng preprocessor.

    Returns:
        Dict[str, Pipeline]: Dictionary chứa các Pipeline chưa được huấn luyện.
    """
    preprocessor = build_classification_preprocessor(X)

    # 1. Pipeline Hồi quy Logic (Logistic Regression)
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

    # 2. Pipeline Hồi quy Tuyến tính (Linear Regression Classifier)
    lin_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearRegressionClassifier()),
        ]
    )

    return {
        "logistic_regression": lr_pipeline,
        "linear_regression": lin_pipeline,
    }


def tune_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Tìm tham số tối ưu nhất cho Hồi quy Logic và Hồi quy Tuyến tính thông qua GridSearchCV.
    Được fit hoàn toàn trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).

    Args:
        X_train (pd.DataFrame): Tập đặc trưng huấn luyện.
        y_train (pd.Series): Nhãn mục tiêu huấn luyện.
        cv (int): Số lượng fold Cross-Validation.
        random_state (int): Random seed phục vụ tái tạo kết quả.

    Returns:
        Dict[str, Dict[str, Any]]: Báo cáo chứa best_estimator_, best_params_ và cv_score.
    """
    pipelines = build_classification_pipelines(X_train)

    # Lưới tham số tối ưu cho Hồi quy Logic
    param_grid_lr = {
        "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__penalty": ["l2"],
    }

    # Lưới tham số tối ưu cho Hồi quy Tuyến tính
    param_grid_lin = {
        "classifier__threshold": [0.3, 0.4, 0.5, 0.6],
        "classifier__fit_intercept": [True, False],
    }

    param_grids = {
        "logistic_regression": param_grid_lr,
        "linear_regression": param_grid_lin,
    }

    tuned_results: Dict[str, Dict[str, Any]] = {}

    for model_name, pipeline in pipelines.items():
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="f1",
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
    Tạo dự đoán nhãn 0/1 trên tập kiểm thử X_test cho các mô hình đã tối ưu tham số.

    Args:
        fitted_models (Dict[str, Any]): Dictionary chứa kết quả từ tune_classifiers.
        X_test (pd.DataFrame): Tập đặc trưng kiểm thử.

    Returns:
        Dict[str, pd.Series]: Nhãn dự đoán tương ứng với index của X_test.
    """
    predictions: Dict[str, pd.Series] = {}

    for model_name, model_info in fitted_models.items():
        if isinstance(model_info, dict) and "estimator" in model_info:
            estimator = model_info["estimator"]
        else:
            estimator = model_info

        y_pred = estimator.predict(X_test)
        predictions[model_name] = pd.Series(
            y_pred, index=X_test.index, name=model_name
        )

    return predictions