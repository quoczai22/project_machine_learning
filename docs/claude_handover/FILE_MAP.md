# Bản đồ file và artifact

## Input

| Đường dẫn | Vai trò | Dùng bởi |
|---|---|---|
| `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dữ liệu Telco gốc | EDA, split classification, luật kết hợp |
| `data/interim/working_raw_7043.csv` | Input clustering 7.043 dòng, chưa scale | `src/clustering.py` |
| `data/processed/Train_Data.csv` | Classification train đã xử lý | `src/experiments.py` |
| `data/processed/Test_Data.csv` | Classification test đã xử lý | `src/experiments.py`, `src/predict.py` |
| `data/processed/classification_split_metadata.json` | Audit split, seed, stratify, fit-on-train | kiểm tra leakage |

## Mã nguồn

| Đường dẫn | Tạo ra |
|---|---|
| `notebooks/EDA.ipynb` | 7 ảnh EDA |
| `src/prepare_classification_data.py` | Train/Test + metadata |
| `src/experiments.py` | classification CSV, confusion matrices, model `.joblib` |
| `src/predict.py` | CSV dự đoán từ model đã lưu |
| `src/clustering.py` | clustering metrics, nhãn cụm, Elbow/Silhouette/PCA |
| `src/association_rules.py` | CSV Apriori/FP-Growth, chart lift cơ bản |
| `scripts/generate_report_figures.py` | classification comparison + cluster profile |
| `scripts/run_association_extension.py` | toàn bộ artifact luật kết hợp dùng Word/PPT |

## Output cần dùng trong Word/PPT

### EDA

`reports/figures/eda/01_...png` đến `07_...png`.

### Classification

- `reports/metrics/classification/classification_results.csv`
- `reports/figures/classification/cm_logistic_regression.png`
- `reports/figures/classification/cm_random_forest_classifier.png`
- `reports/figures/classification/08_classification_comparison.png`
- `models/classification/*.joblib`
- `models/classification/model_manifest.json`

### Clustering

- `reports/metrics/clustering/clustering_metrics.csv`
- `reports/metrics/clustering/algorithm_comparison_k4.csv`
- `reports/metrics/clustering/cluster_summary_k4.csv`
- `reports/metrics/clustering/clustering_result_7043.csv`
- `reports/figures/clustering/elbow_kmeans.png`
- `reports/figures/clustering/silhouette_compare.png`
- `reports/figures/clustering/pca_kmeans_k4.png`
- `reports/figures/clustering/12_cluster_profile.png`

### Luật kết hợp

- `reports/metrics/association_rules/algorithm_comparison.csv`
- `reports/metrics/association_rules/apriori_churn_yes_rules.csv`
- `reports/metrics/association_rules/fpgrowth_churn_yes_rules.csv`
- `reports/metrics/association_rules/association_rules_churn_yes.csv`
- `reports/figures/association_rules/01_association_workflow.png`
- `reports/figures/association_rules/02_algorithm_comparison.png`
- `reports/figures/association_rules/03_rules_support_confidence_lift.png`
- `reports/figures/association_rules/04_top_rules_table.png`

## Không dùng làm input model

- Toàn bộ `reports/`: chỉ là artifact/bằng chứng.
- `reports/qa/`: log và biên bản nghiệm thu.
- `output/`: Word/PPT bàn giao.
- `docs/archive/`: tài liệu cũ.
- `.ppt_build*`, `.docx_render_*`, `.chart-data-*`, `.venv/`, `.pytest_cache/`,
  `logs/`: file tạm/runtime.
