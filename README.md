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
project_machine_learning/
├── data/
│   ├── raw/                         # CSV Telco gốc
│   ├── interim/                     # Input 7.043 dòng cho clustering
│   ├── processed/                   # Train/Test đã tiền xử lý và metadata
│   ├── archive/                     # File bàn giao cũ, chỉ lưu đối chiếu
│   └── README.md                    # Phân biệt dữ liệu chính thức và lưu trữ
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
git clone https://github.com/quoczai22/project_machine_learning.git
cd project_machine_learning
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

Chạy notebook EDA khi cần tái tạo hoặc xem lại quá trình khám phá dữ liệu:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Dữ liệu phân lớp (Classification) đã chốt

Classification dùng cặp file đã tiền xử lý: `data/processed/Train_Data.csv` (5.634 dòng) và `data/processed/Test_Data.csv` (1.409 dòng). Hai file là một cặp split 80/20 đã cố định; `src/experiments.py` dùng trực tiếp cặp này và không split, scale hay one-hot encode lần nữa. GridSearchCV dùng `scoring="f1"`; các model dùng `random_state=42` và `class_weight="balanced"`.

Cặp file này được tạo lại từ raw CSV bằng lệnh sau để có thể kiểm tra quy trình: split 80/20 với `random_state=42`, `stratify=Churn`; imputer, scaler và encoder chỉ `fit` trên Train và chỉ `transform` Test.

```bash
python -m src.prepare_classification_data
python -m src.experiments
python -m pytest tests/test_leakage.py -q
```

Metadata tái tạo nằm tại `data/processed/classification_split_metadata.json`.
Vai trò của từng tệp dữ liệu, gồm cả các tệp lưu đối chiếu không được code sử dụng, được mô tả trong [`data/README.md`](data/README.md).

Luồng xử lý:

```text
data/raw -> prepare_classification_data -> Train_Data/Test_Data -> experiments -> reports/metrics/classification
data/interim/working_raw_7043.csv -> clustering -> reports/metrics/clustering
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

## Các module chính

Các mô hình dùng `random_state=42` để tái tạo kết quả. Dữ liệu phân lớp được chia Train/Test trước khi fit imputer, encoder và scaler để tránh rò rỉ dữ liệu.

| Module | Hàm chính | Vai trò |
|---|---|---|
| `src/data_loader.py` | `load_csv(path)` | Đọc CSV, báo lỗi nếu thiếu file hoặc thiếu cột `customerID`, `TotalCharges`, `Churn`. |
| `src/preprocessor.py` | `clean_telco_data(data)`, `split_features_target(data)`, `build_classification_preprocessor(features)` | Chuyển `TotalCharges` sang số và `Churn` sang 0/1; tách X, y (bỏ `customerID`); tạo bộ tiền xử lý chưa fit (median + StandardScaler cho biến số, One-Hot cho biến phân loại). |
| `src/prepare_classification_data.py` | `prepare_classification_splits(...)` | Chia 80/20 có stratify, fit tiền xử lý chỉ trên Train, xuất `Train_Data.csv`, `Test_Data.csv` và metadata. |
| `src/classification.py` | `build_classification_pipelines(X)`, `tune_classifiers(X_train, y_train)`, `predict_classifiers(models, X_test)` | Pipeline Logistic Regression và Random Forest (`class_weight="balanced"`), GridSearchCV 5-fold với `scoring="f1"`, dự đoán nhãn 0/1. |
| `src/experiments.py` | `load_preprocessed_splits()`, `run_classification_experiments()`, `save_metrics_comparison_plot(...)` | Thực nghiệm cuối trên cặp Train/Test đã chốt; xuất CSV độ đo, ma trận nhầm lẫn và biểu đồ so sánh. Không chia hay tiền xử lý lại. |
| `src/clustering.py` | `load_data(path)`, `evaluate(data)`, `fit_selected(data, k=4)`, `save_plots(...)` | Gom cụm trên `tenure`, `MonthlyCharges`, `service_count` (số dịch vụ có giá trị `Yes`); quét k = 2..10 cho K-Means và Hierarchical; xuất kết quả k = 4 và biểu đồ Elbow, Silhouette, PCA. |
| `src/evaluator.py` | `classification_metrics(y_true, y_pred)`, `clustering_silhouette(features, labels)` | Tính accuracy, precision, recall, F1 cho lớp Churn = 1 và Silhouette Score. |

## Phân công

| Thành viên | Nhiệm vụ | Sản phẩm |
|---|---|---|
| Trịnh Hữu Kiến Quốc (nhóm trưởng) | Nghiệm thu và chốt báo cáo | Rà soát, kiểm thử lại, sửa lỗi tích hợp, đối chiếu số liệu và biểu đồ, chốt báo cáo |
| Đào Sỹ Luân | EDA và khảo sát dữ liệu | `notebooks/EDA.ipynb`, biểu đồ trong `reports/figures/eda/` |
| Nguyễn Tấn Kỳ | Tiền xử lý dữ liệu | `src/data_loader.py`, `src/preprocessor.py`, `src/prepare_classification_data.py` |
| Đoàn Tô Gia Bảo | Phân lớp | `src/classification.py`, `src/experiments.py`, `reports/metrics/classification/` |
| Phạm Khắc Tùng | Gom cụm | `src/clustering.py`, `reports/metrics/clustering/`, `reports/figures/clustering/` |
| Nguyễn Tấn Hiệu | Trực quan hóa và trình bày | Tổng hợp bảng, biểu đồ, caption; slide và phần tổng hợp báo cáo |

## Kiểm thử

```bash
python -m pytest
```

Bộ test kiểm tra tính toàn vẹn dữ liệu, tiền xử lý, chống rò rỉ dữ liệu giữa Train và Test, phân lớp (ma trận nhầm lẫn khớp CSV kết quả), gom cụm và các hàm tính độ đo.
