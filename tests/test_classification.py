"""Unit tests cho src/classification.py.

Đảm bảo kiểm tra dự đoán chi tiết cho cả Logistic Regression và Random Forest.
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.classification import (
    build_classification_pipelines,
    predict_classifiers,
    tune_classifiers,
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 60, 5, 36, 48, 2, 70, 10],
            "MonthlyCharges": [
                29.85,
                56.95,
                53.85,
                42.30,
                70.70,
                99.65,
                89.10,
                29.75,
                104.80,
                20.20,
            ],
            "TotalCharges": [
                29.85,
                1889.50,
                1081.45,
                1840.75,
                151.65,
                3204.40,
                3019.10,
                301.90,
                7261.25,
                202.00,
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
            ],
            "InternetService": [
                "DSL",
                "DSL",
                "DSL",
                "DSL",
                "Fiber optic",
                "Fiber optic",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "No",
            ],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 1, 0, 0, 0, 1], name="Churn")
    return X, y


def test_build_classification_pipelines(sample_data):
    X, _ = sample_data
    pipelines = build_classification_pipelines(X)

    assert "logistic_regression" in pipelines
    assert "random_forest" in pipelines
    assert isinstance(pipelines["logistic_regression"], Pipeline)
    assert isinstance(pipelines["random_forest"], Pipeline)


def test_predict_classifiers_both_models_thoroughly(sample_data):
    X, y = sample_data
    tuned = tune_classifiers(X, y, cv=2)
    preds = predict_classifiers(tuned, X)

    # 1. Test chi tiết Logistic Regression Prediction
    assert "logistic_regression" in preds
    lr_pred = preds["logistic_regression"]
    assert len(lr_pred) == len(X)
    assert set(lr_pred.unique()).issubset({0, 1})

    # 2. Test chi tiết Random Forest Prediction
    assert "random_forest" in preds
    rf_pred = preds["random_forest"]
    assert len(rf_pred) == len(X)
    assert set(rf_pred.unique()).issubset({0, 1})

    # 3. Test Xác suất dự đoán (Predict Proba) cho cả 2
    for model_name in ["logistic_regression", "random_forest"]:
        estimator = tuned[model_name]["estimator"]
        probas = estimator.predict_proba(X)
        assert probas.shape == (len(X), 2)