# Dữ liệu dự án

Thư mục này lưu toàn bộ dữ liệu cần để tái tạo kết quả đã trình bày trong báo cáo. Mã nguồn chỉ dùng các tệp được ghi là **chính thức** dưới đây.

| Thư mục | Tệp chính | Vai trò |
|---|---|---|
| `raw/` | `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dữ liệu Telco Customer Churn gốc gồm 7.043 khách hàng. |
| `processed/` | `Train_Data.csv`, `Test_Data.csv`, `classification_split_metadata.json` | Bộ dữ liệu chính thức cho phân lớp. Tập Train/Test được chia 80/20 với `random_state=42`, `stratify=Churn`; biến đổi chỉ được fit trên Train. |
| `interim/` | `working_raw_7043.csv` | Dữ liệu đầu vào chính thức cho gom cụm. Tệp chưa scale; `src/clustering.py` tự chuẩn hóa ba đặc trưng `tenure`, `MonthlyCharges`, `service_count`. |
| `archive/` | `Full_Clean_Data.csv`, `telco_train_data.csv`, `Clustering_Data_Nguoi4.csv` | Các tệp bàn giao trong quá trình thực hiện, giữ để đối chiếu lịch sử. Code và kết quả cuối **không** dùng các tệp này. |

Không dùng `customerID` làm đặc trưng. Ở bài toán phân lớp, `Churn` là nhãn mục tiêu; ở bài toán gom cụm, cả `customerID` và `Churn` đều được loại khỏi đặc trưng đầu vào.
