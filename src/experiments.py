import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

from preprocessor import clean_telco_data, split_features_target, build_classification_preprocessor
from evaluator import classification_metrics

# 1. Load và tiền xử lý dữ liệu
data_path = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not data_path.exists():
    data_path = Path("data/processed/telco_train_data.csv")

df = pd.read_csv(data_path)
df_clean = clean_telco_data(df)
X, y = split_features_target(df_clean)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# 2. Xây dựng Pipelines
preprocessor = build_classification_preprocessor(X_train)

log_reg = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
rf_clf = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# 3. Huấn luyện và Đánh giá
log_reg.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

metrics_log = classification_metrics(y_test, log_reg.predict(X_test))
metrics_rf = classification_metrics(y_test, rf_clf.predict(X_test))

# 4. Lưu kết quả ra file CSV
results_df = pd.DataFrame([
    {"Model": "Logistic Regression", **metrics_log},
    {"Model": "Random Forest", **metrics_rf}
])

Path("reports").mkdir(exist_ok=True)
results_df.to_csv("reports/classification_results.csv", index=False)
print("--- KẾT QUẢ CLASSIFICATION ---")
print(results_df)