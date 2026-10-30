# Bối cảnh đầy đủ: Bài tập lớn Học máy — Telco Customer Churn

## 1. Mục tiêu bài tập

Đây là bài tập lớn môn Học máy của HUIT, dùng bộ dữ liệu **Telco Customer
Churn** gồm 7.043 khách hàng và 21 cột. Nhóm phải thực hiện ba hướng phân tích
bổ sung nhau:

1. **Phân lớp:** dự đoán khách hàng có rời bỏ dịch vụ hay không (`Churn=Yes/No`).
2. **Gom cụm:** phân khúc khách hàng không giám sát theo đặc trưng hành vi/dịch vụ.
3. **Luật kết hợp:** tìm tổ hợp dịch vụ và bối cảnh xuất hiện cùng `Churn=Yes`.

Phần trình bày/báo cáo cần có EDA, tiền xử lý, so sánh thuật toán, độ đo đánh
giá, trực quan hóa, kết luận và hạn chế. Đây là bài tập học máy truyền thống;
**không dùng Deep Learning**.

## 2. Ranh giới cần giữ đúng

- Không khẳng định luật kết hợp là quan hệ nhân quả. Luật chỉ thể hiện liên hệ
  trong dữ liệu.
- Không dùng `customerID` làm feature cho bất kỳ model hoặc clustering nào.
- `TotalCharges` phải dùng `pd.to_numeric(errors="coerce")`; giá trị không hợp
  lệ được impute median trong bước phù hợp.
- Với classification, split phải xảy ra trước khi fit imputer/scaler/encoder để
  tránh data leakage.
- Luật kết hợp không cần split Train/Test và không đánh giá bằng Accuracy/F1.
- K-Means không bắt buộc số cụm chẵn/lẻ. Việc chọn `k=4` phải giải thích bằng
  Silhouette, Elbow và khả năng diễn giải phân khúc, không dựa vào tính chẵn/lẻ.

## 3. Dữ liệu và hợp đồng đầu vào

### Dữ liệu gốc

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` là CSV gốc, 7.043 dòng.
Không sửa trực tiếp file này.

### Classification

Classification dùng cặp đã chốt:

- `data/processed/Train_Data.csv`: 5.634 dòng.
- `data/processed/Test_Data.csv`: 1.409 dòng.
- `data/processed/classification_split_metadata.json`: bằng chứng quy trình.

Cặp này được tái tạo bằng `src.prepare_classification_data` theo split 80/20,
`random_state=42`, `stratify=Churn`. Imputer, StandardScaler và OneHotEncoder
chỉ `fit` trên Train rồi `transform` Test. CSV đã encode/scale; do đó
`src.experiments.py` **không được encode, scale, impute hoặc split lại**.

### Clustering

`data/interim/working_raw_7043.csv` là input 7.043 dòng, chưa scale, có các
cột `customerID`, `tenure`, `MonthlyCharges`, `service_count`. `src.clustering`
loại `customerID`/`Churn` và fit StandardScaler trong pipeline clustering.

### Luật kết hợp

Luật kết hợp đọc toàn bộ dữ liệu thô. `association_data.py` biến mỗi khách hàng
thành một giao dịch Boolean, giữ item về dịch vụ/bối cảnh; `tenure` được chia
`0-12 tháng`, `13-48 tháng`, `>48 tháng`; `MonthlyCharges` được chia thấp,
trung bình, cao. Chỉ công bố luật có consequent đúng bằng `Churn=Yes`.

## 4. Pipeline và trách nhiệm module

```text
raw CSV
 ├─ EDA notebook → reports/figures/eda/
 ├─ prepare_classification_data → Train_Data/Test_Data + metadata
 │   └─ experiments → classification CSV + confusion matrices + .joblib models
 ├─ working_raw_7043 → clustering → metrics k=2..10 + Elbow/Silhouette/PCA/profile
 └─ association_rules → Apriori/FP-Growth CSV + rule figures
```

| Thành phần | Vai trò |
|---|---|
| `notebooks/EDA.ipynb` | Khám phá dữ liệu và sinh đúng 7 ảnh EDA. |
| `src/data_loader.py` | Đọc CSV và kiểm tra cột tối thiểu. |
| `src/preprocessor.py` | Quy tắc làm sạch và preprocessor dùng trong pipeline tổng quát. |
| `src/prepare_classification_data.py` | Tái tạo cặp Train/Test chống leakage. |
| `src/classification.py` | Pipeline LR/RF tổng quát, GridSearchCV và dự đoán. |
| `src/experiments.py` | Thực nghiệm cuối trên cặp Train/Test đã chốt, xuất metrics/CM/model. |
| `src/predict.py` | Tải `.joblib`, kiểm tra schema bằng manifest, dự đoán CSV đã tiền xử lý. |
| `src/clustering.py` | K-Means và Hierarchical, quét `k=2..10`, chọn `k=4`. |
| `src/association_data.py` | Tạo ma trận giao dịch Boolean. |
| `src/association_rules.py` | Apriori/FP-Growth, lọc consequent `Churn=Yes`. |
| `scripts/generate_report_figures.py` | Sinh biểu đồ so sánh classification và profile cụm. |
| `scripts/run_association_extension.py` | Chạy trọn pipeline luật kết hợp và ảnh báo cáo. |
| `tests/` | Unit tests về tiền xử lý, leakage, classification, clustering, luật kết hợp và prediction. |

## 5. EDA đã có

Các ảnh EDA nằm tại `reports/figures/eda/`:

1. `01_churn_distribution.png`
2. `02_churn_by_contract.png`
3. `03_churn_by_payment_method.png`
4. `04_tenure_distribution_by_churn.png`
5. `05_monthly_charges_by_churn.png`
6. `06_correlation_heatmap.png`
7. `07_churn_by_internet_and_techsupport.png`

Notebook gồm 12 mục; mục 12 kiểm tra đủ 7 ảnh. Mục 8 (tenure) có thêm bảng tỷ
lệ Churn theo `0-12 tháng`/`13-48 tháng`/`>48 tháng` (47,44%/23,64%/9,51%),
cùng cách chia với `tenure_band` của luật kết hợp. Mục 11 (ảnh 07) cho thấy
Fiber optic có tỷ lệ Churn 41,89%, làm nền cho luật kết hợp nổi bật.

EDA có nhiệm vụ mô tả dữ liệu và định hướng feature/tiền xử lý. Nó không được
trộn với script sinh biểu đồ kết quả model. Biểu đồ so sánh classification và
profile clustering được tạo riêng bởi `scripts/generate_report_figures.py`.

## 6. Kết quả classification đã chốt

Từ `reports/metrics/classification/classification_results.csv`:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7424 | 0.5096 | **0.7834** | 0.6175 |
| Random Forest Classifier | **0.7530** | **0.5236** | 0.7701 | **0.6234** |

- Cả hai dùng `class_weight="balanced"`, `random_state=42`.
- GridSearchCV 5-fold dùng `scoring="f1"` trên Train.
- Chọn **Random Forest** làm model chính theo F1-score cao nhất.
- Logistic Regression có Recall cao hơn, nghĩa là bắt được nhiều khách Churn
  hơn nhưng đánh đổi Precision.
- Confusion Matrix cuối phải khớp CSV metrics, được bảo vệ bởi test.
- Mô hình sau khi train được lưu tại `models/classification/`:
  `logistic_regression.joblib`, `random_forest_classifier.joblib`,
  `model_manifest.json`.

## 7. Kết quả clustering đã chốt

Hai thuật toán được so sánh: **K-Means** và **Hierarchical (Ward)**. Quét
`k=2..10` trên cùng ba feature: `tenure`, `MonthlyCharges`, `service_count`.

Các mốc K-Means quan trọng:

| k | Silhouette |
|---:|---:|
| 2 | 0.4255 |
| 3 | 0.4055 |
| 4 | **0.4212** |
| 5 | 0.3835 |
| 6 | 0.3950 |
| 7 | 0.3798 |
| 8 | 0.3678 |
| 9 | 0.3483 |
| 10 | 0.3539 |

K-Means `k=2` đạt Silhouette cao nhất nhưng chỉ tạo hai nhóm tổng quát. Nhóm
chọn **K-Means k=4** vì 0.4212 gần k=2, cao hơn k=3/k=5 lân cận, phù hợp Elbow
và tạo bốn phân khúc khách hàng có thể diễn giải. Tại `k=4`, Hierarchical đạt
Silhouette 0.3457, thấp hơn K-Means.

Artifact chính:

- `reports/figures/clustering/elbow_kmeans.png`
- `reports/figures/clustering/silhouette_compare.png`
- `reports/figures/clustering/pca_kmeans_k4.png`
- `reports/figures/clustering/12_cluster_profile.png`
- `reports/metrics/clustering/cluster_summary_k4.csv`

## 8. Kết quả luật kết hợp đã chốt

Hai thuật toán chạy cùng dữ liệu giao dịch và ngưỡng:

- `min_support=0.05`
- `min_confidence=0.40`
- `min_lift=1.0`
- `max_len=3`

| Algorithm | Frequent itemsets | Luật có Churn=Yes | Runtime hiện tại |
|---|---:|---:|---:|
| Apriori | 6.850 | 127 | khoảng 0.20 giây |
| FP-Growth | 6.850 | 127 | khoảng 4.47 giây |

Runtime phụ thuộc máy/môi trường, vì vậy chỉ diễn giải số frequent itemsets,
số luật và các metric Support/Confidence/Lift là bằng chứng chính. Luật nổi bật:

```text
InternetService=Fiber optic + tenure=0-12 tháng → Churn=Yes
Support 9,16% | Confidence 69,88% | Lift 2,63
```

Diễn giải đúng: trong dữ liệu, nhóm khách Fiber optic mới dùng dịch vụ xuất
hiện Churn cao hơn baseline. Không được nói Fiber optic hoặc tenure ngắn gây
ra Churn.

## 9. Nghiệm thu và lệnh tái tạo

Tại thời điểm bàn giao, full test suite đã chạy **21 passed**.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q

# Tái tạo classification và model đã lưu
python -m src.prepare_classification_data
python -m src.experiments
python -m src.predict --input data/processed/Test_Data.csv

# Tái tạo clustering
python -m src.clustering --data data/interim/working_raw_7043.csv --out reports --k 4

# Sinh ảnh tổng hợp ngoài EDA
python scripts/generate_report_figures.py

# Tái tạo luật kết hợp và ảnh kèm theo
python scripts/run_association_extension.py
```

Sau khi chạy lại, cần đối chiếu CSV metrics với ảnh/Word/PPT; không trộn ảnh
cũ từ lượt chạy khác.

## 10. Báo cáo, PPT và vai trò sáu thành viên

- Báo cáo Word/PPT chỉ dùng artifact sinh từ `reports/` và CSV metrics đã chốt.
- PPT đã có kết luận chung ba hướng: phân lớp dự đoán nguy cơ, gom cụm xác định
  phân khúc, luật kết hợp tìm tổ hợp liên hệ với Churn.
- Phân công học thuật: (1) Data & EDA; (2) preprocessing; (3) classification;
  (4) clustering; (5) evaluation/testing và tổng hợp trực quan; (6) báo cáo.
  Nhóm trưởng theo dõi tiến độ, nghiệm thu, chốt số liệu và sửa lỗi tích hợp.

## 11. Tình trạng repository

- Branch local hiện hành: `recovery/telco-project-cleanup`.
- Project vừa được khôi phục từ worktree bị hỏng; một bản backup còn ở thư mục
  cùng cấp tên `telco-merge-elbow-backup-20261021` để bảo toàn dữ liệu.
- Các thay đổi hiện chưa commit/push. Không dùng `git reset --hard`, không xóa
  backup trước khi người phụ trách kiểm tra và commit xong.
- File build tạm như `.ppt_build*`, `.docx_render_*`, `.chart-data-*` bị ignore.

## 12. Cách trợ lý khác nên hỗ trợ

Ưu tiên kiểm tra số liệu và đường dẫn trước khi đề xuất sửa. Không thay đổi
thuật toán, dữ liệu, số liệu chốt, Word/PPT hoặc branch khi chưa được yêu cầu
rõ. Nếu cần thêm ảnh/bảng, sinh từ script/code hiện hành và cập nhật đồng bộ
toàn bộ artifact liên quan.
