# Cấu trúc project Telco Customer Churn

Tài liệu này mô tả chi tiết cách tổ chức thư mục, luồng dữ liệu (Data Pipeline), mã nguồn và các artifact kết quả của đồ án, giúp dễ dàng truy vết nguồn gốc từ dữ liệu thô đến báo cáo cuối cùng.

---

## 1. Dữ liệu (`data/`)

- **`data/raw/`**:
  - `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dữ liệu gốc từ Telco (7.043 dòng, 21 cột), giữ nguyên bản, không chỉnh sửa trực tiếp.
- **`data/interim/`**:
  - `working_raw_7043.csv`: Dữ liệu trung gian sau khi xử lý missing value ở `TotalCharges` (impute bằng median) và tạo đặc trưng `service_count` phục vụ bài toán Phân cụm (Clustering); chưa scale.
- **`data/processed/`**:
  - `Train_Data.csv` & `Test_Data.csv`: Cặp tập huấn luyện/kiểm tra chuẩn (tỷ lệ 80/20, stratified split theo `Churn`, `random_state=42`) được xử lý theo quy trình chống rò rỉ dữ liệu (data leakage).
  - `classification_split_metadata.json`: Lưu thông số phân chia tập dữ liệu (tỷ lệ, seed, số lượng mẫu).
  - *Các tệp bàn giao thành viên* (`Clustering_Data_Nguoi4.csv`, `Full_Clean_Data.csv`, `telco_train_data.csv`): Được lưu lại để đối chiếu lịch sử phát triển.

---

## 2. Mã nguồn và Kiểm thử

### Mã nguồn chính (`src/`)
- `src/data_loader.py`: Tải và kiểm tra tính hợp lệ của dữ liệu đầu vào.
- `src/preprocessor.py`: Xử lý missing values, mã hóa biến phân loại (encoding), chuẩn hóa đặc trưng (StandardScaler) và trích xuất đặc trưng mới.
- `src/prepare_classification_data.py`: Pipeline tiền xử lý và chia tách Train/Test độc lập, đảm bảo không rò rỉ thông tin từ tập test vào tập train.
- `src/classification.py`: Khai báo và cấu hình các mô hình phân lớp (Logistic Regression, Random Forest), tối ưu siêu tham số bằng GridSearchCV (5-fold CV).
- `src/experiments.py`: Script chạy thực nghiệm phân loại, ghi nhận log và xuất kết quả metrics, Confusion Matrix.
- `src/clustering.py`: Thực hiện phân cụm khách hàng (K-Means, Hierarchical) với $k=2 \dots 10$, đánh giá qua Silhouette/Inertia và xuất kết quả $k=4$.
- `src/evaluator.py`: Tính toán các chỉ số đánh giá mô hình (Accuracy, Precision, Recall, F1-score, ROC-AUC).

### Sổ tay thực nghiệm (`notebooks/`)
- `notebooks/EDA.ipynb`: Khám phá dữ liệu (EDA), trực quan hóa phân bố các biến và phân tích mối quan hệ với Churn.

### Kiểm thử tự động (`tests/`)
- `test_data_integrity.py`: Kiểm tra tính toàn vẹn dữ liệu gốc, schema và các giá trị hợp lệ.
- `test_preprocessor.py`: Kiểm thử các bước tiền xử lý, tính toán đặc trưng `service_count`.
- `test_leakage.py`: Kiểm tra nghiêm ngặt không bị rò rỉ dữ liệu giữa tập Train và Test.
- `test_classification.py`: Kiểm tra pipeline huấn luyện và dự đoán của mô hình phân lớp.
- `test_clustering.py`: Kiểm tra pipeline phân cụm K-Means và Hierarchical.
- `test_evaluator.py`: Kiểm tra độ chính xác của các hàm tính toán metric.

---

## 3. Kết quả đầu ra (`reports/`)

- **Báo cáo nhận xét**:
  - `reports/eda_insights.md`: Nhận xét và đúc kết chi tiết từ quá trình phân tích khám phá dữ liệu (EDA).

- **Biểu đồ trực quan hóa (`reports/figures/`)**:
  - `reports/figures/eda/`: 6 biểu đồ phân tích EDA:
    - `01_churn_distribution.png`: Tỷ lệ Churn tổng thể (26.54% Churn).
    - `02_churn_by_contract.png`: Tỷ lệ Churn theo loại hợp đồng.
    - `03_churn_by_payment_method.png`: Tỷ lệ Churn theo phương thức thanh toán.
    - `04_tenure_distribution_by_churn.png`: Phân bố thời gian gắn bó (tenure).
    - `05_monthly_charges_by_churn.png`: Phân bố chi phí hàng tháng (MonthlyCharges).
    - `06_correlation_heatmap.png`: Ma trận tương quan giữa các biến số.
  - `reports/figures/classification/`:
    - `cm_logistic_regression.png`: Ma trận nhầm lẫn của Logistic Regression.
    - `cm_random_forest_classifier.png`: Ma trận nhầm lẫn của Random Forest.
    - `test_metrics_comparison.png`: Biểu đồ cột so sánh các độ đo Test (Accuracy, Precision, Recall, F1-score) giữa 2 mô hình.
  - `reports/figures/clustering/`:
    - `elbow_kmeans.png`: Đồ thị Elbow phương pháp K-Means ($k=2 \dots 10$).
    - `silhouette_compare.png`: So sánh điểm Silhouette giữa K-Means và Hierarchical.
    - `pca_kmeans_k4.png`: Trực quan hóa 4 cụm khách hàng trên không gian 2D PCA.

- **Bảng số liệu & Metrics (`reports/metrics/`)**:
  - `reports/metrics/classification/`:
    - `classification_results.csv`: Bảng so sánh hiệu năng các mô hình phân loại.
  - `reports/metrics/clustering/`:
    - `clustering_metrics.csv`: Bảng tổng hợp Silhouette score và Inertia cho các giá trị $k=2 \dots 10$.
    - `algorithm_comparison_k4.csv`: So sánh hiệu năng K-Means vs Hierarchical tại $k=4$.
    - `cluster_summary_k4.csv`: Thống kê đặc trưng trung bình của 4 phân khúc khách hàng.
    - `clustering_result_7043.csv`: Dữ liệu 7.043 khách hàng kèm nhãn cụm đã gán.

---

## 4. Nguyên tắc quản lý Artifact

1. **Một chiều (No backward dependency)**: Tuyệt đối không dùng các file sinh ra trong `reports/` làm input để huấn luyện mô hình.
2. **Khả năng tái lập (Reproducibility)**: Toàn bộ bảng biểu và chỉ số trong báo cáo Word phải tái lập được qua các script trong `src/` hoặc notebook trong `notebooks/`.
