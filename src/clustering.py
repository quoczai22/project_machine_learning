import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from pathlib import Path

from preprocessor import clean_telco_data, split_features_target, build_classification_preprocessor
from evaluator import clustering_silhouette

# 1. Load dữ liệu & Biến đổi
data_path = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not data_path.exists():
    data_path = Path("data/telco_train_data.csv")
    if not data_path.exists():
        data_path = Path("data/processed/telco_train_data.csv")

df = pd.read_csv(data_path)
df_clean = clean_telco_data(df)
X, _ = split_features_target(df_clean)

preprocessor = build_classification_preprocessor(X)
X_scaled = preprocessor.fit_transform(X)

# Ép X_scaled về dạng dense numpy array để tránh lỗi SciPy trên Python 3.14
if hasattr(X_scaled, "toarray"):
    X_scaled = X_scaled.toarray()

# 2. Phân cụm (K-Means & Hierarchical với k=2)
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_scaled)
hierarchical = AgglomerativeClustering(n_clusters=2).fit(X_scaled)

score_km = clustering_silhouette(X_scaled, kmeans.labels_)
score_hier = clustering_silhouette(X_scaled, hierarchical.labels_)

print(f"Silhouette Score K-Means (k=2): {score_km:.4f}")
print(f"Silhouette Score Hierarchical (k=2): {score_hier:.4f}")

# 3. Giảm chiều dữ liệu bằng PCA & Vẽ biểu đồ
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.title("PCA K-Means Clustering (k=2)")
plt.savefig("reports/figures/pca_kmeans.png")
plt.close()

print("Đã tạo biểu đồ phân cụm tại reports/figures/pca_kmeans.png")