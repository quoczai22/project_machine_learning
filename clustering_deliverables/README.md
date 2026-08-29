# Clustering deliverable — Telco Customer Churn

## Files

- `clustering.py`: K-Means + Hierarchical (Ward) + DBSCAN benchmark.
- `test_clustering.py`: unit tests for the clustering module.

Copy `clustering.py` into `src/clustering.py`.

## Run

From the repository root:

```bash
python src/clustering.py --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output reports/figures
```

Then inspect:

- `reports/figures/clustering_results.csv`
- `reports/figures/elbow_silhouette.png`
- `reports/figures/pca_kmeans.png`
- `reports/figures/pca_hierarchical.png`
- `reports/figures/pca_dbscan.png`
- `reports/figures/cluster_profiles_kmeans.csv`

Run tests:

```bash
python -m pytest tests/test_clustering.py -q
```

## Method

Clustering uses only:

1. `tenure`
2. `MonthlyCharges`
3. `service_count`

`service_count` counts columns with value exactly `Yes`:
`PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.

`customerID` and `Churn` are never used to fit the clustering models.

Primary model-selection metric: Silhouette Score.

For DBSCAN, noise (`label=-1`) is excluded from the silhouette calculation and
the noise ratio is reported separately.

ARI/NMI against `Churn` are optional external validation metrics. They do not
participate in model fitting.
