# Telco Customer Churn Prediction

Bài tập nhóm môn Học máy - HUIT. Dự án dự đoán khách hàng rời bỏ dịch vụ (`Churn`) và phân khúc khách hàng bằng các thuật toán học máy truyền thống.

## Phạm vi học thuật

- Phân lớp: Logistic Regression và Random Forest, có `class_weight="balanced"`.
- Gom cụm: KMeans và Hierarchical Clustering; số cụm được chọn bằng Elbow và/hoặc Silhouette Score.
- Không sử dụng Deep Learning hay bất kỳ thư viện `tensorflow`, `torch`, `keras`.
- `TotalCharges` phải được chuyển bằng `pd.to_numeric(errors="coerce")` và impute median. `customerID` không được đưa vào feature.
- Tất cả chỉ số trong báo cáo phải được tái tạo từ code và dataset trong `data/`.

## Cấu trúc thư mục

```text
telco-churn-project/
├── data/
│   ├── raw/                         # CSV Telco gốc
│   ├── interim/                     # Input 7.043 dòng cho clustering
│   └── processed/                   # Train/Test đã tiền xử lý và metadata
├── notebooks/
│   └── EDA.ipynb                    # EDA và tạo hình EDA
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── classification.py
│   ├── clustering.py
│   ├── evaluator.py
│   ├── experiments.py
│   └── prepare_classification_data.py
├── tests/                           # 6 bộ unit test kiểm tra chặt chẽ pipeline
├── reports/
│   ├── figures/
│   │   ├── eda/                     # 7 biểu đồ phân bố và tương quan
│   │   ├── classification/          # Confusion Matrix và biểu đồ so sánh độ đo test
│   │   └── clustering/              # Elbow, Silhouette và PCA k=4
│   └── metrics/
│       ├── classification/          # CSV kết quả phân lớp
│       └── clustering/              # CSV Silhouette, Inertia, cluster profile
├── demo/                            # Demo web Streamlit: dự đoán Churn và phân khúc khách hàng
├── requirements.txt
└── README.md
```

## Cài đặt và chạy

Yêu cầu Python 3.10+ (Python 3.8+ được phép theo đề cương, nhưng 3.10+ được khuyến nghị).

```bash
cd telco-churn-project
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài thư viện và chạy test:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

## Dữ liệu phân lớp (Classification) đã chốt

Classification dùng cặp file đã tiền xử lý của thành viên 2: `data/processed/Train_Data.csv` (5.634 dòng) và `data/processed/Test_Data.csv` (1.409 dòng). Hai file là một cặp split 80/20 đã cố định; `src/experiments.py` dùng trực tiếp cặp này và không split, scale hay one-hot encode lần nữa. GridSearchCV dùng `scoring="f1"`; các model dùng `random_state=42` và `class_weight="balanced"`.

Cặp file này được tạo lại từ raw CSV bằng lệnh sau để có thể kiểm tra quy trình: split 80/20 với `random_state=42`, `stratify=Churn`; imputer, scaler và encoder chỉ `fit` trên Train và chỉ `transform` Test.

```bash
python -m src.prepare_classification_data
python -m src.experiments
python -m pytest tests/test_leakage.py -q
```

Metadata tái tạo nằm tại `data/processed/classification_split_metadata.json`.

Khi các module đã được triển khai, pipeline sẽ chạy theo thứ tự:

```text
data_loader -> preprocessor -> classification / clustering -> evaluator
```

## Chạy lại phân cụm (Clustering)

Clustering dùng một bộ input riêng và thống nhất quy trình: quét `k = 2..10` cho K-Means và Hierarchical, sau đó phân tích cuối với `k = 4`. File input `data/interim/working_raw_7043.csv` có 7.043 dòng và chưa bị scale; StandardScaler được fit trong `src/clustering.py`.

```bash
python -m src.clustering --data data/interim/working_raw_7043.csv --out reports --k 4
python -m pytest tests/test_clustering.py -q
```

## Chạy demo web

Thư mục `demo/` có ứng dụng Streamlit gồm hai demo:

1. **Dự đoán Churn**: nhập thông tin khách hàng hoặc lấy một khách thật từ tập Test; Random Forest (mô hình chính) hoặc Logistic Regression trả về xác suất rời bỏ.
2. **Phân khúc khách hàng**: nhập tenure, cước tháng và các dịch vụ đang dùng; K-Means k = 4 xếp khách vào một trong bốn phân khúc, kèm biểu đồ và gợi ý chăm sóc.

```bash
cd demo
python -m streamlit run app.py
```

Trên Windows có thể nhấp đúp `demo/chay_demo.bat`. Trình duyệt mở `http://localhost:8501`.

Lần chạy đầu, `demo/build_artifacts.py` huấn luyện lại mô hình từ `data/raw/` theo đúng quy trình ở trên (split 80/20, chỉ fit trên Train; K-Means k = 4 trên ba đặc trưng đã chuẩn hóa) và dừng nếu số liệu lệch kết quả đã chốt. Mô hình được lưu vào `demo/artifacts/` (không đưa lên Git). Tỷ lệ Churn từng cụm ở demo 2 chỉ tính sau khi gom cụm để diễn giải, không dùng làm đặc trưng.

## Kết quả đã chốt

### Phân lớp (Classification)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7424 | 0.5096 | 0.7834 | 0.6175 |
| Random Forest | 0.7530 | 0.5236 | 0.7701 | 0.6234 |

GridSearchCV dùng 5-fold cross-validation và `scoring="f1"`. Random Forest được chọn theo F1 cao nhất; Logistic Regression có Recall cao hơn.

### Phân cụm (Clustering)

K-Means và Hierarchical được đánh giá với k = 2..10. Silhouette cao nhất của K-Means là k = 2 (0.4255); nhóm chọn K-Means k = 4 (0.4212) để có bốn phân khúc dễ diễn giải hơn. Tại k = 4, Hierarchical đạt 0.3457.

Bằng chứng nghiệm thu clustering và classification được kiểm tra thông qua bộ test trong `tests/`.

## Quy ước artifact

- `reports/figures/eda/`: hình khám phá dữ liệu (01..07).
- `reports/figures/classification/`: Confusion Matrix và hình so sánh độ đo test.
- `reports/figures/clustering/`: Elbow, Silhouette và PCA.
- `reports/metrics/classification/` và `reports/metrics/clustering/`: CSV kết quả phân lớp và phân cụm.

## Hợp đồng interface giữa các module

Tất cả hàm công khai phải có docstring, type hints và không đọc/ghi file ẩn bên trong (trừ các hàm được ghi rõ là I/O). Dùng `random_state=42` mặc định để tái tạo kết quả. Dữ liệu phân lớp phải split trước khi fit imputer/encoder/scaler để tránh data leakage.

### `src/data_loader.py`

```python
from pathlib import Path
import pandas as pd

def load_telco_data(path: str | Path) -> pd.DataFrame:
    """Đọc CSV, kiểm tra các cột bắt buộc và trả về bản sao DataFrame thô."""

def validate_raw_schema(df: pd.DataFrame) -> None:
    """Raise ValueError nếu thiếu cột bắt buộc hoặc dữ liệu rỗng."""
```

Input là CSV gốc có 21 cột; output giữ nguyên `customerID`, `TotalCharges` và `Churn` để preprocessing xử lý tập trung.

### `src/preprocessor.py`

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển TotalCharges sang float, impute median, bỏ customerID và map Churn."""

def split_features_target(
    df: pd.DataFrame, target: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """Trả về X và y nhị phân (No=0, Yes=1), không có customerID."""

def build_classification_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Tạo transformer: numeric Median+StandardScaler, categorical OneHotEncoder."""

def transform_cluster_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, Pipeline]:
    """Tạo feature gom cụm đã scale riêng và trả về scaler/pipeline đã fit."""
```

`clean_telco_data` chỉ làm sạch quy tắc; khi train classification, imputer, encoder và scaler phải nằm trong Pipeline của train fold. Clustering dùng scaler riêng vì tập feature khác classification.

### `src/classification.py`

```python
from sklearn.pipeline import Pipeline
import pandas as pd

def build_classification_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Trả về pipeline logistic_regression và random_forest, đều balanced."""

def tune_classifiers(
    X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5, random_state: int = 42
) -> dict[str, object]:
    """GridSearchCV, fit train-only, trả về best_estimator_ và best_params_ từng model."""

def predict_classifiers(
    fitted_models: dict[str, Pipeline], X_test: pd.DataFrame
) -> dict[str, pd.Series]:
    """Trả về nhãn dự đoán 0/1 của từng model theo index X_test."""
```

`tune_classifiers` không được nhìn `X_test`/`y_test`. Kết quả cần đủ `best_params`, `cv_score` và `estimator` để evaluator tái tạo được báo cáo.

### `src/clustering.py`

```python
import pandas as pd

def build_service_count(df: pd.DataFrame) -> pd.Series:
    """Đếm dịch vụ đăng ký theo quy tắc tài liệu hóa; index giống df."""

def select_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trả về tenure, MonthlyCharges và service_count đã làm sạch, chưa scale."""

def evaluate(data: pd.DataFrame, k_values: range = range(2, 11)):
    """Trả về inertia và silhouette của KMeans và Hierarchical cho mỗi k."""

def fit_selected(data: pd.DataFrame, k: int = 4):
    """Fit KMeans và Hierarchical cho k được chọn."""
```

Số cụm được chọn từ bảng kết quả (không chọn tùy ý). `service_count` cần đếm các dịch vụ có giá trị `Yes`; quy tắc chính thức phải ghi rõ trong docstring và report.

### `src/evaluator.py`

```python
from pathlib import Path
import pandas as pd

def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Tính accuracy, precision, recall, f1 cho lớp Churn=1."""

def clustering_metrics(X_scaled: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Tính silhouette_score; raise ValueError nếu labels không hợp lệ."""

def clustering_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Tính silhouette_score; raise ValueError nếu labels không hợp lệ."""
```

### Tiện ích hỗ trợ (`model_utils`)

Các hàm tiện ích như `ensure_output_dir` và `save_json` được tích hợp trực tiếp trong `src/prepare_classification_data.py` và `src/experiments.py` để quản lý output và metadata JSON.

## Phân công và tiêu chí bàn giao

| Thành viên | Phạm vi | Đầu ra bắt buộc |
|---|---|---|
| 1 - Data & EDA | `notebooks/EDA.ipynb` | Phân bố Churn, Contract, tenure, PaymentMethod và heatmap tương quan |
| 2 - Preprocessing | `data_loader.py`, `preprocessor.py` | Data sạch, schema validation, unit test |
| 3 - Classification | `classification.py` | LR vs RF, GridSearchCV, metrics tái tạo được |
| 4 - Clustering | `clustering.py` | KMeans vs Hierarchical, Elbow/Silhouette, phân khúc |
| 5 - Evaluation & Testing | `evaluator.py`, `tests/` | 100% pytest pass, metric và ảnh Confusion Matrix |
| 6 - Report | `reports/` | Báo cáo chỉ dùng số liệu sinh ra từ pipeline |

## Mốc kiểm tra 14 ngày

| Thời gian | Mốc bàn giao |
|---|---|
| Ngày 1-3 | EDA và code khung module |
| Ngày 4-6 | Data loader/preprocessing; data sạch |
| Ngày 7-10 | Classification, clustering và test |
| Ngày 11-12 | Review chéo; `pytest` pass 100% |
| Ngày 13-14 | Báo cáo và thuyết trình |

## Lệnh review bắt buộc

```bash
python -m pytest
```

Chỉ nhận module khi test tương ứng pass, có docstring, không leakage, và không có import Deep Learning. Dùng mẫu trạng thái trong yêu cầu của trưởng nhóm khi báo cáo review.
