# Nhận xét phân tích dữ liệu Telco Customer Churn

## 1. Tổng quan dữ liệu

Dataset gồm 7.043 khách hàng và 21 cột dữ liệu. Mỗi dòng tương ứng với một khách hàng. Cột `customerID` được sử dụng để định danh khách hàng.

Dữ liệu gồm nhiều thông tin về khách hàng như giới tính, thời gian sử dụng dịch vụ (`tenure`), loại hợp đồng (`Contract`), phương thức thanh toán (`PaymentMethod`), phí hàng tháng (`MonthlyCharges`), tổng số tiền đã thanh toán (`TotalCharges`) và trạng thái rời bỏ dịch vụ (`Churn`).

## 2. Phân bố Churn

Phần lớn khách hàng không rời bỏ dịch vụ, trong khi một phần nhỏ khách hàng có trạng thái Churn = Yes. Tỷ lệ khách hàng Churn khoảng 26,54%, cho thấy vấn đề khách hàng rời bỏ dịch vụ vẫn đáng được quan tâm.

## 3. Churn theo Contract

Nhóm khách hàng sử dụng hợp đồng Month-to-month có tỷ lệ Churn cao nhất, khoảng 42,7%. Trong khi đó, nhóm sử dụng hợp đồng One year có tỷ lệ Churn khoảng 11,3% và nhóm Two year chỉ khoảng 2,8%.

Điều này cho thấy khách hàng sử dụng hợp đồng ngắn hạn có xu hướng rời bỏ dịch vụ cao hơn đáng kể so với khách hàng sử dụng hợp đồng dài hạn.

## 4. Churn theo PaymentMethod

Nhóm khách hàng sử dụng phương thức thanh toán Electronic check có tỷ lệ Churn cao nhất, khoảng 45,3%. Các phương thức thanh toán tự động như Bank transfer (automatic) và Credit card (automatic) có tỷ lệ Churn thấp hơn.

Điều này cho thấy phương thức thanh toán có sự khác biệt đáng kể giữa nhóm khách hàng Churn và không Churn.

## 5. Tenure và Churn

Khách hàng có thời gian sử dụng dịch vụ (`tenure`) ngắn có xu hướng Churn cao hơn. Ngược lại, khách hàng đã sử dụng dịch vụ trong thời gian dài có xu hướng tiếp tục sử dụng dịch vụ và ít rời bỏ hơn.

Điều này cho thấy việc duy trì khách hàng trong những tháng đầu tiên có thể đóng vai trò quan trọng trong việc giảm tỷ lệ Churn.

## 6. MonthlyCharges và Churn

Nhóm khách hàng Churn có xu hướng có mức phí hàng tháng (`MonthlyCharges`) cao hơn nhóm khách hàng không Churn.

Điều này cho thấy mức phí hàng tháng có thể có mối liên hệ với khả năng khách hàng rời bỏ dịch vụ, tuy nhiên đây không phải là yếu tố duy nhất quyết định Churn.

## 7. Heatmap tương quan

Heatmap cho thấy `Churn_num` có tương quan âm với `tenure`, nghĩa là khách hàng sử dụng dịch vụ càng lâu thì xu hướng Churn càng thấp.

`Churn_num` có tương quan dương với `MonthlyCharges`, cho thấy khách hàng có mức phí hàng tháng cao có xu hướng Churn cao hơn.

Ngoài ra, `SeniorCitizen` có tương quan dương với `Churn_num` ở mức khoảng 0,15. Tuy nhiên mức tương quan này không mạnh, vì vậy không thể kết luận SeniorCitizen là yếu tố quyết định trực tiếp việc khách hàng rời bỏ dịch vụ.

## 8. Kết luận

Qua quá trình phân tích, Contract, PaymentMethod, tenure và MonthlyCharges đều cho thấy sự khác biệt giữa nhóm khách hàng Churn và không Churn. Đặc biệt, khách hàng sử dụng hợp đồng Month-to-month và phương thức thanh toán Electronic check có tỷ lệ Churn cao. Khách hàng có tenure thấp cũng có xu hướng rời bỏ dịch vụ nhiều hơn.