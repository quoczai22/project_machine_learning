# Cấu trúc project Telco Customer Churn

Project được tổ chức theo luồng dữ liệu và artifact để người đọc biết file nào là input, file nào là kết quả và file nào là bằng chứng nghiệm thu.

## Dữ liệu

- `data/raw/`: dữ liệu Telco gốc, không chỉnh sửa.
- `data/interim/`: dữ liệu trung gian đã bổ sung `service_count` để chạy clustering; chưa StandardScaler.
- `data/processed/`: cặp `Train_Data.csv` và `Test_Data.csv` đã được tạo theo split tái lập được, cùng metadata. Các file này là input cho classification.

Các CSV thành viên bàn giao còn lại trong `data/processed/` được giữ để truy vết nguồn dữ liệu; chúng không phải là output kết quả cuối nếu không được README gọi trực tiếp.

## Mã nguồn và kiểm thử

- `notebooks/EDA.ipynb`: khám phá dữ liệu và sinh hình EDA.
- `src/prepare_classification_data.py`: tái tạo Train/Test theo quy tắc chống leakage.
- `src/experiments.py`: chạy Logistic Regression và Random Forest, xuất metric và Confusion Matrix.
- `src/clustering.py`: chạy K-Means và Hierarchical với k=2..10, xuất metric và hình clustering.
- `tests/`: kiểm tra tiền xử lý, leakage, classification và clustering.

## Kết quả

- `reports/figures/eda/`: hình phân bố và tương quan.
- `reports/figures/classification/`: Confusion Matrix và hình so sánh phân lớp.
- `reports/figures/clustering/`: Elbow, Silhouette, PCA và profile cụm.
- `reports/metrics/classification/`: CSV metric classification duy nhất.
- `reports/metrics/clustering/`: CSV Silhouette, Inertia, nhãn cụm và profile cụm.

Không dùng file trong `reports/` làm input để train model. Mỗi biểu đồ/bảng trong báo cáo Word phải truy được về một file trong `reports/` và một lệnh chạy ở README.
