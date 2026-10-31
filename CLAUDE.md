# CLAUDE.md: Telco Customer Churn (Nhóm 13, Học máy HUIT)

Bài tập lớn môn Học máy: "Phân tích và dự đoán rời bỏ dịch vụ Telco Customer
Churn" trên bộ dữ liệu Kaggle
https://www.kaggle.com/datasets/blastchar/telco-customer-churn (7.043 dòng × 21 cột).

## Phạm vi (theo đề BT01)

Chỉ làm đúng những gì đề yêu cầu:

| Yêu cầu của đề | Ở đâu trong repo |
|---|---|
| Phân tích, trực quan hóa: phân bố, tương quan | `notebooks/EDA.ipynb` → `reports/figures/eda/01..07`, `reports/eda_insights.md` |
| Tiền xử lý: dữ liệu thiếu, mã hóa, chuẩn hóa | `src/prepare_classification_data.py` |
| 1 thuật toán phân lớp, so sánh với 1 thuật toán khác | Logistic Regression và Random Forest: `src/experiments.py`, `src/classification.py` |
| 1 thuật toán gom cụm, so sánh với 1 thuật toán khác | K-Means và Hierarchical (Ward): `src/clustering.py` |
| Đánh giá: accuracy, precision, recall; silhouette | `reports/metrics/classification/`, `reports/metrics/clustering/` |
| Nhận xét, kết luận | Báo cáo Word và slide (ngoài repo) |

**Ngoài phạm vi, không thêm lại:**

- Luật kết hợp (Apriori, FP-Growth, `mlxtend`): đã bỏ ở PR #12 vì đề không yêu cầu.
- Deep Learning (`tensorflow`, `torch`, `keras`).
- Thuật toán, đặc trưng hay hướng phân tích mới khi chưa được nhóm trưởng yêu cầu.

F1-score, ma trận nhầm lẫn, GridSearchCV, Elbow và PCA được giữ vì chúng phục vụ
trực tiếp các yêu cầu trên, không phải hướng làm thêm.

## Quy tắc dữ liệu bắt buộc

- `customerID` không bao giờ là feature. `Churn` là nhãn phân lớp và không dùng để gom cụm.
- `TotalCharges` chuyển bằng `pd.to_numeric(errors="coerce")`. Có 11 ô khoảng trắng, đều
  là khách `tenure = 0`; điền bằng median tính trên Train.
- Phân lớp: chia 80/20, `random_state=42`, `stratify=Churn`. Imputer, scaler và encoder
  (`OneHotEncoder(drop="first", handle_unknown="ignore")`) chỉ fit trên Train.
- `data/processed/Train_Data.csv` và `Test_Data.csv` đã được mã hóa và chuẩn hóa sẵn.
  `src/experiments.py` không được chia hay tiền xử lý lại.
- Gom cụm dùng `data/interim/working_raw_7043.csv` (chưa chuẩn hóa) với 3 đặc trưng
  `tenure`, `MonthlyCharges`, `service_count` (số cột có giá trị "Yes" trong 8 cột dịch vụ).
  StandardScaler fit bên trong `src/clustering.py`.
- `Clustering_Data_Nguoi4.csv`, `Full_Clean_Data.csv`, `telco_train_data.csv` chỉ là file
  lịch sử của thành viên, không dùng làm input.

## Số liệu đã chốt (không được thay đổi hay bịa)

| Mô hình | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (C=0.1, lbfgs) | 0.7424 | 0.5096 | 0.7834 | 0.6175 |
| Random Forest (100 cây, max_depth=10), mô hình chính | 0.7530 | 0.5236 | 0.7701 | 0.6234 |

- Cả hai dùng `class_weight="balanced"`, GridSearchCV 5-fold với `scoring="f1"`.
- K-Means k=4 (`random_state=42`, `n_init=20`): Silhouette 0.4212; Hierarchical k=4: 0.3457.
- K-Means k=2 có Silhouette cao nhất (0.4255). Chọn k=4 vì sát mức đó, cao hơn k=3 và
  k=5, phù hợp Elbow và cho bốn phân khúc dễ diễn giải. **Không** nói "k=4 cao nhất" và
  không chọn k=4 vì là số chẵn.
- Bốn cụm K-Means có 1.557 / 2.144 / 2.315 / 1.027 khách.

Mọi số liệu phải lấy từ CSV trong `reports/metrics/` hoặc chạy lại code. Nếu lệch, báo
file nguồn và lệnh chạy lại; không sửa số bằng tay.

## Lệnh chạy

```bash
python -m pip install -r requirements.txt
python -m pytest -q                      # hiện tại: 16 passed
python -m src.prepare_classification_data
python -m src.experiments
python -m src.clustering --data data/interim/working_raw_7043.csv --out reports --k 4
```

## Quy trình Git

- `main` yêu cầu Pull Request và 1 lượt duyệt. Làm trên nhánh mới, không push thẳng `main`.
- Không dùng `git reset --hard` hay xóa backup khi chưa có xác nhận của nhóm trưởng.
- Mỗi thay đổi cần chạy `pytest` và báo "Đạt / Chưa đạt" kèm bằng chứng.
