"""Unit tests cho src/classification.py.

Đảm bảo kiểm tra dự đoán chi tiết cho cả Logistic Regression và Random Forest.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from src.classification import (
    build_classification_pipelines,
    predict_classifiers,
    tune_classifiers,
)
from src.experiments import load_preprocessed_splits


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
    tuned = tune_classifiers(
        X,
        y,
        cv=2,
        param_grids={
            "logistic_regression": {
                "classifier__C": [1.0],
                "classifier__solver": ["lbfgs"],
            },
            "random_forest": {
                "classifier__n_estimators": [10],
                "classifier__max_depth": [3],
                "classifier__min_samples_split": [2],
            },
        },
    )
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


def test_final_confusion_matrices_match_classification_results():
    """Lock the final Train/Test confusion matrices to the approved CSV metrics."""
    X_train, X_test, y_train, y_test = load_preprocessed_splits()
    results = pd.read_csv(
        Path("reports/metrics/classification/classification_results.csv")
    ).set_index("Model")

    models = {
        "Logistic Regression": (
            LogisticRegression(
                C=0.1,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            ),
            [[3054, 1085], [296, 1199]],
            [[753, 282], [81, 293]],
        ),
        "Random Forest Classifier": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
            ),
            [[3364, 775], [65, 1430]],
            [[773, 262], [86, 288]],
        ),
    }

    for model_name, (model, expected_train_cm, expected_test_cm) in models.items():
        model.fit(X_train, y_train)
        train_cm = confusion_matrix(y_train, model.predict(X_train))
        test_predictions = model.predict(X_test)
        test_cm = confusion_matrix(y_test, test_predictions)

        assert train_cm.tolist() == expected_train_cm
        assert test_cm.tolist() == expected_test_cm

        row = results.loc[model_name]
        assert accuracy_score(y_test, test_predictions) == pytest.approx(
            row["Accuracy"], abs=5e-5
        )
        assert precision_score(y_test, test_predictions) == pytest.approx(
            row["Precision"], abs=5e-5
        )
        assert recall_score(y_test, test_predictions) == pytest.approx(
            row["Recall"], abs=5e-5
        )
