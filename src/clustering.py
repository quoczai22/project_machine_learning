"""Clustering benchmark for the Telco Customer Churn project.

Required deliverable:
- K-Means vs Hierarchical Clustering on the real Telco CSV.
- Elbow + Silhouette evaluation.
- PCA plots.
- Explicit optimal k for each algorithm.
- Business-readable cluster profiles.
- Reproducible logging throughout the run.
- Optional DBSCAN benchmark as an extension.

Clustering NEVER uses Churn or customerID as model inputs.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "service_count"]
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clustering")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def build_service_count(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in SERVICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing service columns: {missing}")
    return df[SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype(float)


def select_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"tenure", "MonthlyCharges", *SERVICE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing clustering columns: {missing}")
    out = pd.DataFrame(index=df.index)
    out["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    out["service_count"] = build_service_count(df)
    if out.isna().any().any():
        raise ValueError("Clustering features contain missing/non-numeric values")
    return out


def scale_cluster_features(features: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[CLUSTER_FEATURES])
    return pd.DataFrame(scaled, index=features.index, columns=CLUSTER_FEATURES), scaler


def silhouette_or_nan(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(X, labels))


def evaluate_kmeans_hierarchical(X_scaled: pd.DataFrame, k_values=range(2, 9)) -> pd.DataFrame:
    rows = []
    X = X_scaled.to_numpy()
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        km_labels = km.fit_predict(X)
        rows.append({
            "algorithm": "KMeans", "k": k, "parameters": f"k={k}",
            "inertia": float(km.inertia_), "silhouette": silhouette_or_nan(X, km_labels),
            "noise_ratio": 0.0,
        })

        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hc_labels = hc.fit_predict(X)
        rows.append({
            "algorithm": "Hierarchical", "k": k, "parameters": f"k={k}, linkage=ward",
            "inertia": float("nan"), "silhouette": silhouette_or_nan(X, hc_labels),
            "noise_ratio": 0.0,
        })
    return pd.DataFrame(rows)


def evaluate_dbscan(X_scaled: pd.DataFrame,
                    eps_values=(0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.20),
                    min_samples_values=(5, 10, 15, 20)) -> pd.DataFrame:
    rows = []
    X = X_scaled.to_numpy()
    for eps in eps_values:
        for ms in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            non_noise = labels != -1
            valid = labels[non_noise]
            n_clusters = len(np.unique(valid)) if len(valid) else 0
            sil = silhouette_or_nan(X[non_noise], valid) if n_clusters >= 2 else float("nan")
            rows.append({
                "algorithm": "DBSCAN", "k": n_clusters,
                "parameters": f"eps={eps}, min_samples={ms}",
                "inertia": float("nan"), "silhouette": sil,
                "noise_ratio": float((~non_noise).mean()),
                "eps": eps, "min_samples": ms,
            })
    return pd.DataFrame(rows)



# Backward-compatible names used by the existing project tests.
evaluate_cluster_candidates = evaluate_kmeans_hierarchical
evaluate_dbscan_candidates = evaluate_dbscan

def add_external_metrics(results: pd.DataFrame, X_scaled: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
    if y is None:
        return results
    y_arr = np.asarray(y).astype(int)
    X = X_scaled.to_numpy()
    aris, nmis = [], []
    for r in results.itertuples(index=False):
        if r.algorithm == "KMeans":
            labels = KMeans(n_clusters=int(r.k), random_state=RANDOM_STATE, n_init=20).fit_predict(X)
            mask = np.ones(len(labels), dtype=bool)
        elif r.algorithm == "Hierarchical":
            labels = AgglomerativeClustering(n_clusters=int(r.k), linkage="ward").fit_predict(X)
            mask = np.ones(len(labels), dtype=bool)
        else:
            labels = DBSCAN(eps=float(r.eps), min_samples=int(r.min_samples)).fit_predict(X)
            mask = labels != -1
        if mask.sum() > 1 and len(np.unique(labels[mask])) >= 2:
            aris.append(adjusted_rand_score(y_arr[mask], labels[mask]))
            nmis.append(normalized_mutual_info_score(y_arr[mask], labels[mask]))
        else:
            aris.append(float("nan")); nmis.append(float("nan"))
    out = results.copy()
    out["ARI_vs_Churn"] = aris
    out["NMI_vs_Churn"] = nmis
    return out


def rank_results(results: pd.DataFrame):
    ranked = results.sort_values(["silhouette", "noise_ratio"], ascending=[False, True], na_position="last").reset_index(drop=True)
    selected = {}
    for alg in ["KMeans", "Hierarchical", "DBSCAN"]:
        sub = ranked[(ranked.algorithm == alg) & ranked.silhouette.notna()]
        if not sub.empty:
            selected[alg] = sub.iloc[0].to_dict()
    return ranked, selected


def save_elbow_silhouette(results: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    km = results[results.algorithm == "KMeans"].sort_values("k")
    axes[0].plot(km.k, km.inertia, marker="o")
    axes[0].set(title="K-Means Elbow", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[0].grid(alpha=.25)
    for alg in ["KMeans", "Hierarchical"]:
        sub = results[results.algorithm == alg].sort_values("k")
        axes[1].plot(sub.k, sub.silhouette, marker="o", label=alg)
    axes[1].set(title="Silhouette by k", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def save_pca(X_scaled: pd.DataFrame, labels: np.ndarray, title: str, path: Path):
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled.to_numpy())
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=12, alpha=.65)
    ax.set(title=title, xlabel="PCA 1", ylabel="PCA 2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    p = df.copy(); p["Cluster"] = labels
    p["service_count"] = build_service_count(p)
    p["tenure"] = pd.to_numeric(p["tenure"], errors="coerce")
    p["MonthlyCharges"] = pd.to_numeric(p["MonthlyCharges"], errors="coerce")
    rows = []
    for cluster, g in p.groupby("Cluster"):
        row = {
            "Cluster": int(cluster), "Customers": len(g),
            "CustomerPct": round(100 * len(g) / len(p), 2),
            "AvgTenureMonths": round(g.tenure.mean(), 2),
            "AvgMonthlyCharges": round(g.MonthlyCharges.mean(), 2),
            "AvgServiceCount": round(g.service_count.mean(), 2),
        }
        if "Churn" in g.columns:
            row["ChurnRatePct"] = round(100 * g.Churn.eq("Yes").mean(), 2)
        if "Contract" in g.columns:
            row["TopContract"] = g.Contract.mode().iat[0] if not g.Contract.mode().empty else ""
        if "InternetService" in g.columns:
            row["TopInternetService"] = g.InternetService.mode().iat[0] if not g.InternetService.mode().empty else ""
        if "PaymentMethod" in g.columns:
            row["TopPaymentMethod"] = g.PaymentMethod.mode().iat[0] if not g.PaymentMethod.mode().empty else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Cluster")


def save_business_summary(profile: pd.DataFrame, path: Path):
    lines = ["# Business interpretation of clusters", ""]
    for _, r in profile.iterrows():
        churn = f", churn {r['ChurnRatePct']:.2f}%" if "ChurnRatePct" in r else ""
        lines.append(
            f"- Cluster {int(r.Cluster)}: {int(r.Customers)} customers ({r.CustomerPct:.2f}%), "
            f"average tenure {r.AvgTenureMonths:.2f} months, "
            f"monthly charge {r.AvgMonthlyCharges:.2f}, "
            f"average services {r.AvgServiceCount:.2f}{churn}. "
            f"Typical contract: {r.get('TopContract','N/A')}; "
            f"internet: {r.get('TopInternetService','N/A')}; "
            f"payment: {r.get('TopPaymentMethod','N/A')}."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_clustering(csv_path: str | Path, output_dir: str | Path = "reports/figures"):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output / "clustering_run.log")
    logger.info("START clustering pipeline")
    logger.info("Input: %s", csv_path)
    logger.info("Output: %s", output)

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows x %d columns", *df.shape)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        logger.info("TotalCharges converted to numeric; not used as clustering feature")

    features = select_cluster_features(df)
    logger.info("Features: %s", ", ".join(CLUSTER_FEATURES))
    logger.info("Churn/customerID excluded from model fitting")
    logger.info("service_count mean=%.2f, min=%d, max=%d", features.service_count.mean(), int(features.service_count.min()), int(features.service_count.max()))
    X_scaled, _ = scale_cluster_features(features)
    logger.info("StandardScaler fitted")

    logger.info("Evaluating K-Means + Hierarchical for k=2..8")
    results = evaluate_kmeans_hierarchical(X_scaled)
    logger.info("Evaluating DBSCAN parameter grid as extension")
    results = pd.concat([results, evaluate_dbscan(X_scaled)], ignore_index=True)
    y = df["Churn"].map({"No": 0, "Yes": 1}) if "Churn" in df.columns else None
    results = add_external_metrics(results, X_scaled, y)
    ranked, selected = rank_results(results)
    ranked.to_csv(output / "clustering_results.csv", index=False)

    logger.info("BEST MODELS")
    for alg, row in selected.items():
        logger.info("%s -> %s | silhouette=%.4f | noise=%.2f%%", alg, row["parameters"], row["silhouette"], 100*row.get("noise_ratio", 0))
    logger.info("Overall best by silhouette: %s | %s", ranked.iloc[0].algorithm, ranked.iloc[0].parameters)

    save_elbow_silhouette(results, output / "elbow_silhouette.png")
    logger.info("Saved elbow/silhouette figure")

    for alg, fname in [("KMeans", "pca_kmeans.png"), ("Hierarchical", "pca_hierarchical.png"), ("DBSCAN", "pca_dbscan.png")]:
        if alg not in selected: continue
        r = selected[alg]
        if alg == "KMeans":
            labels = KMeans(n_clusters=int(r["k"]), random_state=RANDOM_STATE, n_init=20).fit_predict(X_scaled)
        elif alg == "Hierarchical":
            labels = AgglomerativeClustering(n_clusters=int(r["k"]), linkage="ward").fit_predict(X_scaled)
        else:
            labels = DBSCAN(eps=float(r["eps"]), min_samples=int(r["min_samples"])).fit_predict(X_scaled)
        save_pca(X_scaled, labels, f"{alg} ({r['parameters']})", output / fname)
        profile = profile_clusters(df, labels)
        profile.to_csv(output / f"cluster_profiles_{alg.lower()}.csv", index=False)
        save_business_summary(profile, output / f"cluster_business_{alg.lower()}.md")
        logger.info("Saved PCA + business profile for %s", alg)

    # Human-readable summary
    summary = output / "clustering_summary.md"
    lines = ["# Clustering summary", "", f"Input: `{csv_path}`", f"Rows: {len(df)}", "", "## Best configuration by algorithm", ""]
    for alg, r in selected.items():
        lines.append(f"- **{alg}**: {r['parameters']}; Silhouette = **{r['silhouette']:.4f}**; noise = {100*r.get('noise_ratio',0):.2f}%.")
        if "ARI_vs_Churn" in r and pd.notna(r["ARI_vs_Churn"]):
            lines.append(f"  - ARI vs Churn = {r['ARI_vs_Churn']:.4f}; NMI = {r['NMI_vs_Churn']:.4f}.")
    lines += ["", f"## Overall best by Silhouette", f"**{ranked.iloc[0].algorithm} — {ranked.iloc[0].parameters}** with Silhouette **{ranked.iloc[0].silhouette:.4f}**.", "", "Silhouette is the primary model-selection metric; Churn is used only for external validation/business profiling."]
    summary.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved final summary")
    logger.info("END clustering pipeline")
    return ranked, selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output", default="reports/figures")
    args = parser.parse_args()
    results, selected = run_clustering(args.data, args.output)
    print("\n=== BEST CONFIGURATION BY ALGORITHM ===")
    for alg, r in selected.items():
        print(f"{alg:12s} | {r['parameters']:32s} | silhouette={r['silhouette']:.4f}")
    print("\n=== TOP 10 ===")
    cols = ["algorithm","parameters","silhouette","noise_ratio","ARI_vs_Churn","NMI_vs_Churn"]
    print(results[[c for c in cols if c in results.columns]].head(10).to_string(index=False))
