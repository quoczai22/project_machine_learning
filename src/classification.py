
"""Module classification.py.

Xây dựng, tối ưu hóa siêu tham số (GridSearchCV) và dự đoán nhãn phân lớp
cho 2 mô hình: Logistic Regression và Random Forest Classifier.

"""

from typing import Any, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessor import build_classification_preprocessor


def build_classification_pipelines(X: pd.DataFrame) -> Dict[str, Pipeline]:
    """Tạo dictionary chứa các Pipeline chưa fit cho Logistic Regression và

    Random Forest.

    Cả hai mô hình đều bắt buộc thiết lập class_weight="balanced".

    Args:
        X (pd.DataFrame): Tập đặc trưng đầu vào để xây dựng preprocessor.

    Returns:
        Dict[str, Pipeline]: Dictionary chứa các Pipeline chưa qua huấn luyện.
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

    # 2. Pipeline Random Forest Classifier
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
    """Tìm tham số tối ưu cho Logistic Regression và Random Forest bằng

    GridSearchCV.

    Chỉ fit trên X_train và y_train để tránh rò rỉ dữ liệu (Data Leakage).

    Args:
        X_train (pd.DataFrame): Tập đặc trưng huấn luyện.
        y_train (pd.Series): Nhãn mục tiêu huấn luyện (0/1).
        cv (int): Số fold Cross-Validation.
        random_state (int): Seed phục vụ tái tạo kết quả.

    Returns:
        Dict[str, Dict[str, Any]]: Báo cáo chứa best_estimator_, best_params_
        và cv_score.
    """
    pipelines = build_classification_pipelines(X_train)

    # Lưới tham số tối ưu cho Logistic Regression
    param_grid_lr = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    # Lưới tham số tối ưu cho Random Forest
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
    """Tạo dự đoán nhãn 0/1 trên tập kiểm thử X_test cho các mô hình đã tối ưu.

    Args:
        fitted_models (Dict[str, Any]): Dictionary chứa kết quả từ
          tune_classifiers.
        X_test (pd.DataFrame): Tập đặc trưng kiểm thử.

    Returns:
        Dict[str, pd.Series]: Nhãn dự đoán dạng pd.Series giữ nguyên index của
        X_test.
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