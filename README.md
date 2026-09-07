# Telco Customer Churn Prediction

Bai tap nhom mon Hoc may - HUIT. Du an du doan khach hang roi bo dich vu
(`Churn`) va phan khuc khach hang bang cac thuat toan hoc may truyen thong.

## Pham vi hoc thuat

- Phan lop: Logistic Regression va Random Forest, co `class_weight="balanced"`.
- Gom cum: KMeans va Hierarchical Clustering; so cum duoc chon bang Elbow va/
  hoac Silhouette Score.
- Khong su dung Deep Learning hay bat ky thu vien `tensorflow`, `torch`, `keras`.
- `TotalCharges` phai duoc chuyen bang `pd.to_numeric(errors="coerce")` va
  impute median. `customerID` khong duoc dua vao feature.
- Tat ca chi so trong bao cao phai duoc tai tao tu code va dataset trong `data/`.

## Cau truc

```text
telco-churn-project/
├── data/
│   ├── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── working_raw_7043.csv
│   └── processed/Train_Data.csv, Test_Data.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── classification.py
│   ├── clustering.py
│   ├── evaluator.py
│   └── model_utils.py
├── tests/
├── notebooks/EDA.ipynb
├── reports/
│   └── figures/
├── requirements.txt
└── README.md
```

## Cai dat va chay

Yeu cau Python 3.10+ (Python 3.8+ duoc phep theo de cuong, nhung 3.10+ duoc
khuyen nghi).

```bash
cd telco-churn-project
python -m venv .venv
```

Kich hoat moi truong ao tren Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cai thu vien va chay test:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

## Du lieu classification da chot

Classification dung cap file da tien xu ly cua thanh vien 2:
`data/processed/Train_Data.csv` (5.634 dong) va
`data/processed/Test_Data.csv` (1.409 dong). Hai file la mot cap split 80/20
da co dinh; `src/experiments.py` dung truc tiep cap nay va khong split, scale
hay one-hot encode lan nua. GridSearchCV dung `scoring="f1"`; cac model dung
`random_state=42` va `class_weight="balanced"`.

Khi cac module da duoc trien khai, pipeline se chay theo thu tu:

```text
data_loader -> preprocessor -> classification / clustering -> evaluator
```

## Chay lai clustering

Clustering dung mot bo input rieng va thong nhat quy trinh: quet `k = 2..6`
cho K-Means va Hierarchical, sau do phan tich cuoi voi `k = 4`. File input
`data/working_raw_7043.csv` co 7.043 dong va chua bi scale; StandardScaler
duoc fit trong `src/clustering.py`.

```bash
python -m src.clustering --data data/working_raw_7043.csv --out reports --k 4
python -m pytest tests/test_clustering.py -q
```

## Ket qua da chot

### Classification

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7424 | 0.5096 | 0.7834 | 0.6175 |
| Random Forest | 0.7530 | 0.5236 | 0.7701 | 0.6234 |

GridSearchCV dung 5-fold cross-validation va `scoring="f1"`. Random Forest
duoc chon theo F1 cao nhat; Logistic Regression co Recall cao hon.

### Clustering

K-Means va Hierarchical duoc danh gia voi k = 2..6. Silhouette cao nhat cua
K-Means la k = 2 (0.4255); nhom chon K-Means k = 4 (0.4212) de co bon phan
khuc de dien giai hon. Tai k = 4, Hierarchical dat 0.3457.

Bang chung nghiem thu clustering nam tai `reports/qa/NGHIEM_THU_CLUSTERING.md`;
log test ghi nhan `2 passed` nam tai `reports/qa/pytest_clustering.txt`.

## Hop dong interface giua cac module

Tat ca ham cong khai phai co docstring, type hints va khong doc/ghi file an
ben trong (tru cac ham duoc ghi ro la I/O). Dung `random_state=42` mac dinh de
tai tao ket qua. Du lieu phan lop phai split truoc khi fit imputer/encoder/scaler
de tranh data leakage.

### `src/data_loader.py`

```python
from pathlib import Path
import pandas as pd

def load_telco_data(path: str | Path) -> pd.DataFrame:
    """Doc CSV, kiem tra cac cot bat buoc va tra ve ban sao DataFrame thô."""

def validate_raw_schema(df: pd.DataFrame) -> None:
    """Raise ValueError neu thieu cot bat buoc hoac du lieu rong."""
```

Input la CSV goc co 21 cot; output giu nguyen `customerID`, `TotalCharges` va
`Churn` de preprocessing xu ly tap trung.

### `src/preprocessor.py`

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyen TotalCharges sang float, impute median, bo customerID va map Churn."""

def split_features_target(
    df: pd.DataFrame, target: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """Tra ve X va y nhi phan (No=0, Yes=1), khong co customerID."""

def build_classification_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Tao transformer: numeric Median+StandardScaler, categorical OneHotEncoder."""

def transform_cluster_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, Pipeline]:
    """Tao feature gom cum da scale rieng va tra ve scaler/pipeline da fit."""
```

`clean_telco_data` chi lam sach quy tac; khi train classification, imputer,
encoder va scaler phai nam trong Pipeline cua train fold. Clustering dung scaler
rieng vi tap feature khac classification.

### `src/classification.py`

```python
from sklearn.pipeline import Pipeline
import pandas as pd

def build_classification_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Tra ve pipeline logistic_regression va random_forest, deu balanced."""

def tune_classifiers(
    X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5, random_state: int = 42
) -> dict[str, object]:
    """GridSearchCV, fit train-only, tra ve best_estimator_ va best_params_ tung model."""

def predict_classifiers(
    fitted_models: dict[str, Pipeline], X_test: pd.DataFrame
) -> dict[str, pd.Series]:
    """Tra ve nhan du doan 0/1 cua tung model theo index X_test."""
```

`tune_classifiers` khong duoc nhin `X_test`/`y_test`. Ket qua can du `best_params`,
`cv_score` va `estimator` de evaluator tai tao duoc bao cao.

### `src/clustering.py`

```python
import pandas as pd

def build_service_count(df: pd.DataFrame) -> pd.Series:
    """Dem dich vu dang ky theo quy tac tai lieu hoa; index giong df."""

def select_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tra ve tenure, MonthlyCharges va service_count da lam sach, chua scale."""

def evaluate(data: pd.DataFrame, k_values: range = range(2, 7)):
    """Tra ve inertia va silhouette cua KMeans va Hierarchical cho moi k."""

def fit_selected(data: pd.DataFrame, k: int = 4):
    """Fit KMeans va Hierarchical cho k duoc chon."""
```

So cum duoc chon tu bang ket qua (khong chon tuy y). `service_count` can dem cac
dich vu co gia tri `Yes`; quy tac chinh thuc phai ghi ro trong docstring va report.

### `src/evaluator.py`

```python
from pathlib import Path
import pandas as pd

def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Tinh accuracy, precision, recall, f1 cho lop Churn=1."""

def clustering_metrics(X_scaled: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Tinh silhouette_score; raise ValueError neu labels khong hop le."""

def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str, output_path: str | Path) -> None:
    """Luu anh confusion matrix tai output_path."""

def save_feature_importance(fitted_pipeline: object, output_path: str | Path, top_n: int = 15) -> None:
    """Luu anh feature importance cua Random Forest tai output_path."""
```

### `src/model_utils.py`

```python
from pathlib import Path
import json

RANDOM_STATE = 42

def ensure_output_dir(path: str | Path) -> Path:
    """Tao va tra ve thu muc output."""

def save_json(payload: dict, output_path: str | Path) -> None:
    """Luu metrics/metadata theo JSON co the tai tao."""
```

## Phan cong va tieu chi ban giao

| Thanh vien | Pham vi | Dau ra bat buoc |
|---|---|---|
| 1 - Data & EDA | `notebooks/EDA.ipynb` | Phan bo Churn, Contract, tenure, PaymentMethod va heatmap tuong quan |
| 2 - Preprocessing | `data_loader.py`, `preprocessor.py` | Data sach, schema validation, test don vi |
| 3 - Classification | `classification.py` | LR vs RF, GridSearchCV, metrics tai tao duoc |
| 4 - Clustering | `clustering.py` | KMeans vs Hierarchical, Elbow/Silhouette, phan khuc |
| 5 - Evaluation & Testing | `evaluator.py`, `tests/` | 100% pytest pass, anh CM va feature importance |
| 6 - Report | `report/` | Bao cao chi dung so lieu sinh ra tu pipeline |

## Moc kiem tra 14 ngay

| Thoi gian | Moc ban giao |
|---|---|
| Ngay 1-3 | EDA va code khung module |
| Ngay 4-6 | Data loader/preprocessing; data sach |
| Ngay 7-10 | Classification, clustering va test |
| Ngay 11-12 | Review cheo; `pytest` pass 100% |
| Ngay 13-14 | Bao cao va thuyet trinh |

## Lenh review bat buoc

```bash
python -m pytest
```

Chi nhan module khi test tuong ung pass, co docstring, khong leakage, va khong
co import Deep Learning. Dung mau trang thai trong yeu cau cua truong nhom khi
bao cao review.
