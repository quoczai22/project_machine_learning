"""Classification experiment using member 2's paired train/test datasets.

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


if __name__ == "__main__":
    run_classification_experiments()
