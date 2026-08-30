# Nhận xét phân tích dữ liệu Telco Customer Churn

## 1. Tổng quan dữ liệu

Dataset gồm **7.043 khách hàng** và **21 cột dữ liệu**. Mỗi dòng tương ứng với một khách hàng. Cột `customerID` được sử dụng để định danh khách hàng.

Trong quá trình tiền xử lý, `TotalCharges` được chuyển từ kiểu `object` sang kiểu số (`float64`). Có **11 giá trị trống** trong `TotalCharges`; thay vì xóa 11 dòng, các giá trị này được **impute bằng median = 1.397,475**. Vì vậy, dữ liệu sau xử lý vẫn giữ nguyên **7.043 dòng** và không còn missing values.

## 2. Phân bố Churn

Trong **7.043 khách hàng**, có **5.174 khách hàng không Churn (73,46%)** và **1.869 khách hàng Churn (26,54%)**.

Phần lớn khách hàng không rời bỏ dịch vụ, nhưng nhóm Churn vẫn chiếm hơn một phần tư tổng số khách hàng. Vì vậy, việc xác định các yếu tố liên quan đến Churn là cần thiết.

## 3. Churn theo Contract

Nhóm **Month-to-month** có tỷ lệ Churn cao nhất, khoảng **42,71%**. Nhóm **One year** có tỷ lệ Churn khoảng **11,27%**, còn **Two year** chỉ khoảng **2,83%**.

Điều này cho thấy khách hàng sử dụng hợp đồng ngắn hạn có xu hướng rời bỏ dịch vụ cao hơn đáng kể so với khách hàng ký hợp đồng dài hạn.

## 4. Churn theo PaymentMethod

Nhóm sử dụng **Electronic check** có tỷ lệ Churn cao nhất, khoảng **45,29%**. Các nhóm còn lại có tỷ lệ thấp hơn: **Mailed check 19,11%**, **Bank transfer (automatic) 16,71%** và **Credit card (automatic) 15,24%**.

Electronic check vì vậy là nhóm đáng được chú ý khi phân tích nguyên nhân Churn.

## 5. Tenure và Churn

Median `tenure` của nhóm Churn là **10 tháng**, trong khi nhóm không Churn là **38 tháng**.

Điều này cho thấy khách hàng mới sử dụng dịch vụ có xu hướng Churn cao hơn. Giai đoạn đầu của vòng đời khách hàng có thể là thời điểm cần tăng cường chăm sóc và các biện pháp giữ chân.

## 6. MonthlyCharges và Churn

Median `MonthlyCharges` của nhóm Churn là khoảng **79,65**, cao hơn nhóm không Churn khoảng **64,43**.

Điều này cho thấy khách hàng có mức phí hàng tháng cao có xu hướng Churn nhiều hơn trong bộ dữ liệu. Tuy nhiên, đây chỉ là mối liên hệ quan sát được và chưa thể kết luận về quan hệ nhân quả.

## 7. Heatmap tương quan

Trong nhóm biến số được xét, `tenure` có tương quan âm với `Churn_num` (**r ≈ -0,35**), cho thấy khách hàng có thời gian sử dụng lâu hơn có xu hướng Churn thấp hơn.

`MonthlyCharges` có tương quan dương với `Churn_num` (**r ≈ 0,19**), trong khi `TotalCharges` có tương quan âm (**r ≈ -0,20**). `SeniorCitizen` có tương quan dương nhưng khá yếu với `Churn_num` (**r ≈ 0,15**).

Các hệ số tương quan chỉ phản ánh mức độ liên hệ tuyến tính giữa các biến và **không chứng minh quan hệ nhân quả**.

## 8. Kết luận

Qua phân tích **7.043 khách hàng**, các biến `Contract`, `PaymentMethod`, `tenure` và `MonthlyCharges` đều cho thấy sự khác biệt giữa nhóm Churn và không Churn.

Đáng chú ý nhất:
- **Month-to-month** có tỷ lệ Churn cao nhất (**42,71%**).
- **Electronic check** có tỷ lệ Churn cao nhất trong các phương thức thanh toán (**45,29%**).
- Median `tenure` của nhóm Churn (**10 tháng**) thấp hơn nhiều so với nhóm không Churn (**38 tháng**).
- Median `MonthlyCharges` của nhóm Churn (**79,65**) cao hơn nhóm không Churn (**64,43**).

Các kết quả này có thể được sử dụng làm cơ sở cho bước tiếp theo là xây dựng mô hình dự đoán Churn và đánh giá các yếu tố ảnh hưởng đến khả năng khách hàng rời bỏ dịch vụ.
