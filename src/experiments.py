import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessor import clean_telco_data, split_features_target, build_classification_preprocessor

# 1. Đọc dữ liệu
data_path = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not data_path.exists():
    data_path = Path("data/telco_train_data.csv")
    if not data_path.exists():
        data_path = Path("data/processed/telco_train_data.csv")

df = pd.read_csv(data_path)
df_clean = clean_telco_data(df)
X, y = split_features_target(df_clean)

# 2. Chia dữ liệu & Tiền xử lý
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = build_classification_preprocessor(X_train)
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# 3. Huấn luyện mô hình
log_reg = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_trans, y_train)
rf_cls = RandomForestClassifier(random_state=42).fit(X_train_trans, y_train)

# 4. Đánh giá kết quả bằng sklearn.metrics
def get_metrics(y_true, y_pred):
    pos_val = 1 if 1 in y_true.values else y_true.unique()[1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=pos_val),
        "Recall": recall_score(y_true, y_pred, pos_label=pos_val),
        "F1-score": f1_score(y_true, y_pred, pos_label=pos_val)
    }

res_log = get_metrics(y_test, log_reg.predict(X_test_trans))
res_rf = get_metrics(y_test, rf_cls.predict(X_test_trans))

results_df = pd.DataFrame([
    {"Model": "Logistic Regression", **res_log},
    {"Model": "Random Forest", **res_rf}
])

print("--- KẾT QUẢ CLASSIFICATION ---")
print(results_df)

# Ghi kết quả vào data/processed/classification_results.csv
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_dir / "classification_results.csv", index=False)
print(f"\nĐã lưu kết quả vào {output_dir / 'classification_results.csv'}")

# 5. Vẽ biểu đồ Learning Curve (Kiểm tra Overfitting / Underfitting)
def plot_learning_curves(estimator, X_data, y_data, title, filename):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_data, y_data, cv=5, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score (F1)')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score (F1)')
    
    plt.title(f'Learning Curve: {title}')
    plt.xlabel('Training Examples')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / filename)
    plt.close()
    print(f"Đã lưu biểu đồ Learning Curve tại {fig_dir / filename}")

# Tiến hành biến đổi toàn bộ X và vẽ biểu đồ cho 2 mô hình
X_all_trans = preprocessor.fit_transform(X)
plot_learning_curves(log_reg, X_all_trans, y, "Logistic Regression", "learning_curve_log_reg.png")
plot_learning_curves(rf_cls, X_all_trans, y, "Random Forest", "learning_curve_random_forest.png")