"""Customer segmentation with K-Means, Hierarchical Clustering and DBSCAN.

This module is intentionally independent from the classification pipeline.
The clustering features are:
    - tenure
    - MonthlyCharges
    - service_count

``Churn`` and ``customerID`` are never used to fit a clustering model.
``Churn`` can be supplied only after fitting for an optional external
evaluation (ARI/NMI) and for business profiling.

Service-count rule:
    Count a service when its value is exactly ``"Yes"``.
    ``No``, ``No internet service`` and ``No phone service`` contribute 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "service_count"]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def build_service_count(df: pd.DataFrame) -> pd.Series:
    """Count subscribed services whose value is exactly ``Yes``.

    Args:
        df: Cleaned Telco customer DataFrame.

    Returns:
        Series aligned with ``df.index`` containing the number of active
        services for each customer.

    Raises:
        ValueError: If a required service column is missing.
    """
    missing = [c for c in SERVICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing service columns: {missing}")

    return df[SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype(float)


def select_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the three documented clustering features.

    ``TotalCharges`` and ``Churn`` are deliberately excluded. ``customerID``
    is also excluded because it is only an identifier.
    """
    required = {"tenure", "MonthlyCharges", *SERVICE_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing clustering columns: {missing}")

    features = pd.DataFrame(index=df.index)
    features["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    features["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"], errors="coerce"
    )
    features["service_count"] = build_service_count(df)

    if features.isna().any().any():
        raise ValueError("Clustering features contain missing/non-numeric values.")

    return features


def scale_cluster_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale clustering features and return the fitted scaler."""
    missing = [c for c in CLUSTER_FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[CLUSTER_FEATURES])
    scaled_df = pd.DataFrame(
        scaled, index=features.index, columns=CLUSTER_FEATURES
    )
    return scaled_df, scaler


def _silhouette_or_nan(X: np.ndarray, labels: np.ndarray) -> float:
    """Return silhouette when at least two non-empty clusters exist."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(X, labels))


def evaluate_cluster_candidates(
    X_scaled: pd.DataFrame,
    k_values: range = range(2, 9),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate K-Means and Hierarchical candidates for every k.

    Returns a tidy DataFrame with inertia (K-Means only) and silhouette.
    Hierarchical uses Ward linkage because the clustering space is
    standardized and Euclidean distance is appropriate.
    """
    rows: list[dict[str, Any]] = []
    X = X_scaled.to_numpy()

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )
        km_labels = kmeans.fit_predict(X)
        rows.append(
            {
                "algorithm": "KMeans",
                "parameters": f"k={k}",
                "k": k,
                "linkage": None,
                "eps": None,
                "min_samples": None,
                "inertia": float(kmeans.inertia_),
                "silhouette": _silhouette_or_nan(X, km_labels),
                "noise_ratio": 0.0,
            }
        )

        hierarchical = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )
        h_labels = hierarchical.fit_predict(X)
        rows.append(
            {
                "algorithm": "Hierarchical",
                "parameters": f"k={k}, linkage=ward",
                "k": k,
                "linkage": "ward",
                "eps": None,
                "min_samples": None,
                "inertia": float("nan"),
                "silhouette": _silhouette_or_nan(X, h_labels),
                "noise_ratio": 0.0,
            }
        )

    return pd.DataFrame(rows)


def evaluate_dbscan_candidates(
    X_scaled: pd.DataFrame,
    eps_values: tuple[float, ...] = (
        0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.20
    ),
    min_samples_values: tuple[int, ...] = (5, 10, 15, 20),
) -> pd.DataFrame:
    """Evaluate DBSCAN over a small reproducible parameter grid.

    Noise points (label ``-1``) are excluded from the silhouette calculation,
    while their proportion is reported separately. Configurations with fewer
    than two non-noise clusters receive NaN silhouette.
    """
    rows: list[dict[str, Any]] = []
    X = X_scaled.to_numpy()

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

            non_noise = labels != -1
            n_noise = int((~non_noise).sum())
            noise_ratio = n_noise / len(labels)

            valid_labels = labels[non_noise]
            if (
                valid_labels.size > 0
                and len(np.unique(valid_labels)) >= 2
                and len(valid_labels) > len(np.unique(valid_labels))
            ):
                silhouette = float(
                    silhouette_score(X[non_noise], valid_labels)
                )
            else:
                silhouette = float("nan")

            rows.append(
                {
                    "algorithm": "DBSCAN",
                    "parameters": (
                        f"eps={eps}, min_samples={min_samples}"
                    ),
                    "k": len(set(valid_labels)) if valid_labels.size else 0,
                    "linkage": None,
                    "eps": eps,
                    "min_samples": min_samples,
                    "inertia": float("nan"),
                    "silhouette": silhouette,
                    "noise_ratio": noise_ratio,
                }
            )

    return pd.DataFrame(rows)


def fit_cluster_models(
    X_scaled: pd.DataFrame,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Fit the required K-Means and Hierarchical models at chosen k.

    The selected k should come from ``evaluate_cluster_candidates`` rather
    than being chosen arbitrarily.
    """
    X = X_scaled.to_numpy()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )

    return {
        "kmeans": {
            "model": kmeans,
            "labels": kmeans.fit_predict(X),
        },
        "hierarchical": {
            "model": hierarchical,
            "labels": hierarchical.fit_predict(X),
        },
    }


def fit_dbscan(
    X_scaled: pd.DataFrame,
    eps: float,
    min_samples: int,
) -> dict[str, object]:
    """Fit DBSCAN using a parameter combination selected by evaluation."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled.to_numpy())
    return {"model": model, "labels": labels}


def compare_best_models(
    X_scaled: pd.DataFrame,
    y_churn: pd.Series | None = None,
    k_values: range = range(2, 9),
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Run the complete benchmark and return a ranked result table.

    The primary ranking criterion is silhouette score. If ``y_churn`` is
    provided, ARI/NMI are added as external validation only; the labels are
    never used to fit any model.
    """
    classical = evaluate_cluster_candidates(X_scaled, k_values=k_values)
    dbscan = evaluate_dbscan_candidates(X_scaled)
    results = pd.concat([classical, dbscan], ignore_index=True)

    if y_churn is not None:
        y = np.asarray(y_churn).astype(int)
        ari_values = []
        nmi_values = []

        for row in results.itertuples(index=False):
            if row.algorithm == "KMeans":
                model = KMeans(
                    n_clusters=int(row.k),
                    random_state=RANDOM_STATE,
                    n_init=20,
                )
                labels = model.fit_predict(X_scaled.to_numpy())
                mask = np.ones(len(labels), dtype=bool)
            elif row.algorithm == "Hierarchical":
                model = AgglomerativeClustering(
                    n_clusters=int(row.k), linkage="ward"
                )
                labels = model.fit_predict(X_scaled.to_numpy())
                mask = np.ones(len(labels), dtype=bool)
            else:
                model = DBSCAN(
                    eps=float(row.eps),
                    min_samples=int(row.min_samples),
                )
                labels = model.fit_predict(X_scaled.to_numpy())
                mask = labels != -1

            if mask.sum() > 1 and len(np.unique(labels[mask])) >= 2:
                ari_values.append(adjusted_rand_score(y[mask], labels[mask]))
                nmi_values.append(
                    normalized_mutual_info_score(y[mask], labels[mask])
                )
            else:
                ari_values.append(float("nan"))
                nmi_values.append(float("nan"))

        results["ARI_vs_Churn"] = ari_values
        results["NMI_vs_Churn"] = nmi_values

    ranked = results.sort_values(
        by=["silhouette", "noise_ratio"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    selected: dict[str, dict[str, object]] = {}
    for algorithm in ranked["algorithm"].unique():
        subset = ranked[ranked["algorithm"] == algorithm].dropna(
            subset=["silhouette"]
        )
        if subset.empty:
            continue
        best = subset.iloc[0]
        selected[algorithm] = best.to_dict()

    return ranked, selected


def save_elbow_silhouette_plot(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save Elbow and Silhouette curves for K-Means/Hierarchical."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    km = results[results["algorithm"] == "KMeans"].sort_values("k")
    axes[0].plot(km["k"], km["inertia"], marker="o")
    axes[0].set_title("K-Means Elbow")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.25)

    for algorithm in ["KMeans", "Hierarchical"]:
        subset = results[results["algorithm"] == algorithm].sort_values("k")
        axes[1].plot(
            subset["k"],
            subset["silhouette"],
            marker="o",
            label=algorithm,
        )
    axes[1].set_title("Silhouette by k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_pca_clusters(
    X_scaled: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Save a 2D PCA visualization of one clustering result."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        X_scaled.to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=12,
        alpha=0.65,
    )
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Create a business-readable cluster profile.

    If Churn is present, the table includes churn rate. Churn is not used
    during clustering.
    """
    profile = df.copy()
    profile["Cluster"] = labels

    agg: dict[str, tuple[str, str]] = {
        "Customers": ("Cluster", "size"),
        "AvgTenure": ("tenure", "mean"),
        "AvgMonthlyCharges": ("MonthlyCharges", "mean"),
    }

    if "Churn" in profile.columns:
        if profile["Churn"].dtype == object:
            churn = profile["Churn"].map({"No": 0, "Yes": 1})
        else:
            churn = pd.to_numeric(profile["Churn"], errors="coerce")
        profile["_churn_binary"] = churn
        agg["ChurnRate"] = ("_churn_binary", "mean")

    result = profile.groupby("Cluster").agg(**agg).reset_index()
    if "ChurnRate" in result:
        result["ChurnRate"] *= 100
    return result


def run_clustering(
    csv_path: str | Path,
    output_dir: str | Path = "reports/figures",
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Run the complete clustering experiment from the raw Telco CSV.

    Saves:
        clustering_results.csv
        cluster_profiles_kmeans.csv
        elbow_silhouette.png
        pca_kmeans.png
        pca_hierarchical.png
        pca_dbscan.png
    """
    csv_path = Path(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # TotalCharges is not used for clustering, but converting it here keeps
    # the raw-data handling consistent with the project contract.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    features = select_cluster_features(df)
    X_scaled, _ = scale_cluster_features(features)

    y = None
    if "Churn" in df.columns:
        y = df["Churn"].map({"No": 0, "Yes": 1})

    results, selected = compare_best_models(X_scaled, y_churn=y)
    results.to_csv(output / "clustering_results.csv", index=False)

    save_elbow_silhouette_plot(results, output / "elbow_silhouette.png")

    # K-Means best model
    km = selected.get("KMeans")
    if km is not None:
        km_model = KMeans(
            n_clusters=int(km["k"]),
            random_state=RANDOM_STATE,
            n_init=20,
        )
        km_labels = km_model.fit_predict(X_scaled.to_numpy())
        save_pca_clusters(
            X_scaled,
            km_labels,
            f"K-Means (k={int(km['k'])})",
            output / "pca_kmeans.png",
        )
        km_profile = profile_clusters(df, km_labels)
        km_profile.to_csv(
            output / "cluster_profiles_kmeans.csv", index=False
        )

    # Hierarchical best model
    hc = selected.get("Hierarchical")
    if hc is not None:
        hc_model = AgglomerativeClustering(
            n_clusters=int(hc["k"]),
            linkage="ward",
        )
        hc_labels = hc_model.fit_predict(X_scaled.to_numpy())
        save_pca_clusters(
            X_scaled,
            hc_labels,
            f"Hierarchical (k={int(hc['k'])}, linkage=ward)",
            output / "pca_hierarchical.png",
        )

    # DBSCAN best model
    db = selected.get("DBSCAN")
    if db is not None:
        db_model = DBSCAN(
            eps=float(db["eps"]),
            min_samples=int(db["min_samples"]),
        )
        db_labels = db_model.fit_predict(X_scaled.to_numpy())
        save_pca_clusters(
            X_scaled,
            db_labels,
            f"DBSCAN (eps={db['eps']}, min_samples={int(db['min_samples'])})",
            output / "pca_dbscan.png",
        )

    return results, selected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Telco clustering algorithms."
    )
    parser.add_argument(
        "--data",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/figures",
    )
    args = parser.parse_args()

    results, selected = run_clustering(args.data, args.output)

    print("\n=== BEST CONFIGURATION BY ALGORITHM ===")
    for algorithm, row in selected.items():
        print(
            f"{algorithm:12s} | {row['parameters']:35s} | "
            f"silhouette={row['silhouette']:.4f}"
        )

    print("\n=== TOP 10 CONFIGURATIONS ===")
    columns = [
        "algorithm",
        "parameters",
        "silhouette",
        "noise_ratio",
        "ARI_vs_Churn",
        "NMI_vs_Churn",
    ]
    available = [c for c in columns if c in results.columns]
    print(results[available].head(10).to_string(index=False))
