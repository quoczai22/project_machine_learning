# Nhận xét phân tích dữ liệu Telco Customer Churn

## 1. Tổng quan dữ liệu

Dataset gồm 7.043 khách hàng và 21 cột dữ liệu. Mỗi dòng tương ứng với một khách hàng. Cột `customerID` được sử dụng để định danh khách hàng.

Dữ liệu gồm nhiều thông tin về khách hàng như giới tính, thời gian sử dụng dịch vụ (`tenure`), loại hợp đồng (`Contract`), phương thức thanh toán (`PaymentMethod`), phí hàng tháng (`MonthlyCharges`), tổng số tiền đã thanh toán (`TotalCharges`) và trạng thái rời bỏ dịch vụ (`Churn`).

Các giá trị thiếu trong `TotalCharges` đã được chuyển sang kiểu số và điền bằng median, thay vì loại bỏ 11 dòng, nên toàn bộ 7.043 khách hàng được giữ nguyên trong phân tích.

## 2. Phân bố Churn

Trong toàn bộ 7.043 khách hàng, phần lớn khách hàng không rời bỏ dịch vụ, trong khi một phần đáng kể vẫn có trạng thái Churn = Yes. Khi giữ nguyên toàn bộ dữ liệu sau khi impute median cho `TotalCharges`, phân tích cho thấy tỉ lệ khách hàng Churn vẫn là một vấn đề đáng quan tâm và cần tiếp tục theo dõi.

## 3. Churn theo Contract

Khi phân tích trên toàn bộ 7.043 khách hàng, nhóm khách hàng sử dụng hợp đồng Month-to-month vẫn có tỷ lệ Churn cao hơn đáng kể so với các nhóm hợp đồng dài hạn. Điều này cho thấy loại hợp đồng ngắn hạn có mối liên hệ trực tiếp với nguy cơ khách hàng rời bỏ dịch vụ.

## 4. Churn theo PaymentMethod

Trên toàn bộ dataset không loại bỏ hàng nào, nhóm khách hàng sử dụng phương thức thanh toán Electronic check vẫn có tỷ lệ Churn cao hơn rõ rệt so với các phương thức thanh toán tự động. Điều này cho thấy phương thức thanh toán có thể là một dấu hiệu cảnh báo đáng chú ý trong việc dự đoán nguy cơ rời bỏ dịch vụ.

## 5. Tenure và Churn

Khách hàng có thời gian sử dụng dịch vụ (`tenure`) ngắn vẫn có xu hướng Churn cao hơn. Ngược lại, khách hàng ở giai đoạn sử dụng lâu dài thường tiếp tục sử dụng dịch vụ và ít rời bỏ hơn. Kết quả này được giữ nguyên khi áp dụng median imputation cho `TotalCharges` mà không bỏ bất kỳ dòng nào.

## 6. MonthlyCharges và Churn

Nhóm khách hàng Churn vẫn có xu hướng có mức phí hàng tháng (`MonthlyCharges`) cao hơn nhóm không Churn khi phân tích trên 7.043 khách hàng đầy đủ. Điều này cho thấy mức phí hàng tháng có thể là một yếu tố liên quan đến nguy cơ rời bỏ dịch vụ, nhưng không phải là yếu tố duy nhất.

## 7. Heatmap tương quan

Heatmap trên tập dữ liệu đầy đủ 7.043 khách hàng vẫn cho thấy `Churn_num` có tương quan âm với `tenure`, nghĩa là khách hàng sử dụng dịch vụ lâu thì xu hướng Churn giảm. `Churn_num` lại có tương quan dương với `MonthlyCharges`, phản ánh rằng mức phí hàng tháng cao hơn có thể đi kèm với nguy cơ rời bỏ dịch vụ cao hơn.

Mặc dù `SeniorCitizen` có một mức tương quan nhỏ với `Churn_num`, giá trị này không mạnh nên không thể kết luận đây là yếu tố quyết định độc lập. Tất cả nhận xét trên đều được tính trên dữ liệu đầy đủ sau khi impute median cho `TotalCharges` mà không bỏ hàng nào.

## 8. Kết luận

Qua quá trình phân tích, Contract, PaymentMethod, tenure và MonthlyCharges đều cho thấy sự khác biệt giữa nhóm khách hàng Churn và không Churn. Đặc biệt, khách hàng sử dụng hợp đồng Month-to-month và phương thức thanh toán Electronic check có tỷ lệ Churn cao. Khách hàng có tenure thấp cũng có xu hướng rời bỏ dịch vụ nhiều hơn.