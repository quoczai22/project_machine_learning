"""Customer clustering: K-Means vs Hierarchical.

Input must be the original 7,043-row CSV containing raw numeric columns:
customerID, tenure, MonthlyCharges, service_count.
Churn is deliberately ignored and customerID is never used as a feature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["tenure", "MonthlyCharges", "service_count"]
ALGORITHMS = ("KMeans", "Hierarchical")


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"customerID", *FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    if len(df) != 7043:
        raise ValueError(f"CSV phải có 7,043 dòng, nhưng nhận được {len(df):,} dòng")
    data = df[["customerID", *FEATURES]].copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[FEATURES].isna().any().any():
        raise ValueError("FEATURES có giá trị không phải số hoặc bị thiếu")
    # Guard against accidentally feeding the old already-scaled/encoded CSV.
    if (
        data["tenure"].between(-2.5, 2.5).all()
        and abs(data["tenure"].mean()) < 0.1
        and abs(data["tenure"].std() - 1) < 0.1
    ):
        raise ValueError(
            "File đầu vào có vẻ đã StandardScaler (tenure ~ mean 0, std 1). "
            "Hãy dùng CSV gốc, chưa scale/encode."
        )
    return data


def evaluate(data: pd.DataFrame, k_values=range(2, 7), random_state=42):
    X = StandardScaler().fit_transform(data[FEATURES])
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels_km = km.fit_predict(X)
        rows.append({
            "algorithm": "KMeans", "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels_km),
        })
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_hc = hc.fit_predict(X)
        rows.append({
            "algorithm": "Hierarchical", "k": k,
            "inertia": np.nan,
            "silhouette": silhouette_score(X, labels_hc),
        })
    return X, pd.DataFrame(rows)


def fit_selected(data: pd.DataFrame, k=4, random_state=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[FEATURES])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    out = data.copy()
    out["cluster_kmeans"] = km.fit_predict(X)
    out["cluster_hierarchical"] = hc.fit_predict(X)
    comparison = pd.DataFrame({
        "Algorithm": ["K-Means", "Hierarchical"],
        "k": [k, k],
        "Silhouette": [
            silhouette_score(X, out["cluster_kmeans"]),
            silhouette_score(X, out["cluster_hierarchical"]),
        ],
    })
    return out, comparison, X


def save_plots(metrics, X, labels, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    km = metrics[metrics.algorithm == "KMeans"]
    hc = metrics[metrics.algorithm == "Hierarchical"]

    plt.figure(figsize=(7, 5))
    plt.plot(km.k, km.inertia, marker="o")
    plt.xticks(km.k)
    plt.xlabel("Số cụm k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method - K-Means")
    plt.tight_layout()
    plt.savefig(out_dir / "elbow_kmeans.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(km.k, km.silhouette, marker="o", label="K-Means")
    plt.plot(hc.k, hc.silhouette, marker="s", label="Hierarchical")
    plt.xticks(km.k)
    plt.xlabel("Số cụm k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score: K-Means vs Hierarchical")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "silhouette_compare.png", dpi=180)
    plt.close()

    pca = PCA(n_components=2)
    xp = pca.fit_transform(X)
    plt.figure(figsize=(7, 5))
    for c in sorted(np.unique(labels)):
        mask = labels == c
        plt.scatter(xp[mask, 0], xp[mask, 1], s=10, label=f"Cụm {c}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means với k=4 (PCA 2D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_kmeans_k4.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Đường dẫn CSV gốc 7,043 dòng")
    parser.add_argument("--out", default="reports", help="Thư mục lưu kết quả")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    X, metrics = evaluate(data)
    result, comparison, X = fit_selected(data, k=args.k)

    metrics.to_csv(out_dir / "clustering_metrics.csv", index=False)
    result.to_csv(out_dir / "clustering_result_7043.csv", index=False)
    comparison.to_csv(out_dir / "algorithm_comparison_k4.csv", index=False)

    summary_rows = []
    for alg, col in (("KMeans", "cluster_kmeans"), ("Hierarchical", "cluster_hierarchical")):
        for cluster, g in result.groupby(col):
            summary_rows.append({
                "algorithm": alg,
                "cluster": int(cluster),
                "count": len(g),
                "tenure_mean": g.tenure.mean(),
                "MonthlyCharges_mean": g.MonthlyCharges.mean(),
                "service_count_mean": g.service_count.mean(),
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "cluster_summary_k4.csv", index=False)
    save_plots(metrics, X, result["cluster_kmeans"].to_numpy(), out_dir)

    print("Hoàn tất clustering.")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
