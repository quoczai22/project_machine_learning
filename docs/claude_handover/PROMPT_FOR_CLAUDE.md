# Prompt dán cho Claude

```text
Bạn đang hỗ trợ một nhóm sinh viên hoàn thiện bài tập lớn môn Học máy của HUIT:
“Phân tích và dự đoán rời bỏ dịch vụ Telco Customer Churn”. Hãy xem đây là
ngữ cảnh dự án đã có sẵn; không tự suy diễn thành đề tài mới, không thay đổi
file/code/Word/PPT/Git nếu tôi chưa yêu cầu rõ.

THƯ MỤC PROJECT CẦN LÀM VIỆC:
C:\Users\kienq\OneDrive - University Of Industry and Trade\Tài liệu\ChatGPT\Machine-learning\telco-merge-elbow

ĐƯỢC GIAO VIỆC ĐẦU TIÊN:
1) Đi tới đúng thư mục project ở trên.
2) Đọc hồ sơ bàn giao và kiểm tra cấu trúc/file hiện có.
3) Xác nhận lại bằng một bản tóm tắt ngắn: mục tiêu bài, ba hướng phân tích,
   dữ liệu đầu vào của từng hướng, số liệu đã chốt, branch Git hiện tại.
4) Chạy `python -m pytest -q` nếu môi trường đã có dependencies. Nếu thiếu
   dependencies, chỉ báo rõ thiếu gì; không tự sửa thuật toán hoặc thay số liệu.
5) Sau khi nghiệm thu, chờ yêu cầu cụ thể tiếp theo của nhóm trưởng.

Trước tiên, đọc các file sau trong repository:
1) docs/claude_handover/PROJECT_CONTEXT.md
2) docs/claude_handover/FILE_MAP.md
3) README.md

Mục tiêu học thuật gồm ba hướng bổ sung:
- Phân lớp: Logistic Regression và Random Forest để dự đoán Churn Yes/No.
- Gom cụm: K-Means và Hierarchical, quét k=2..10, chốt K-Means k=4.
- Luật kết hợp: Apriori và FP-Growth, chỉ xét luật có consequent Churn=Yes.

Quy tắc bắt buộc:
- Không dùng Deep Learning.
- customerID không được dùng làm feature.
- TotalCharges chuyển bằng pd.to_numeric(errors="coerce"); xử lý thiếu phù hợp.
- Với classification: split 80/20, random_state=42, stratify=Churn; các bước
  impute/scale/encode chỉ fit trên Train để tránh leakage.
- Train_Data.csv/Test_Data.csv đã là dữ liệu encode/scale. src.experiments.py
  không được split hoặc preprocess lại.
- Luật kết hợp chỉ là liên hệ thống kê, không phải quan hệ nhân quả.
- Không chọn k=4 chỉ vì là số chẵn; bảo vệ bằng Silhouette, Elbow và ý nghĩa
  bốn phân khúc.

Số liệu đã chốt:
- Logistic Regression: Accuracy 0.7424, Precision 0.5096, Recall 0.7834,
  F1 0.6175.
- Random Forest: Accuracy 0.7530, Precision 0.5236, Recall 0.7701,
  F1 0.6234. Chọn đây là model chính theo F1.
- K-Means k=4: Silhouette 0.4212. K=2 đạt 0.4255 nhưng quá tổng quát; k=4
  gần mức đó và tạo bốn phân khúc dễ diễn giải. Hierarchical k=4 đạt 0.3457.
- Apriori và FP-Growth: cùng 6.850 frequent itemsets, 127 luật Churn=Yes.
  Luật tiêu biểu: Fiber optic + tenure 0–12 tháng → Churn=Yes, Support 9,16%,
  Confidence 69,88%, Lift 2,63.
- Full test suite đã từng đạt 21 passed.

Khi trả lời:
1) Luôn đối chiếu code, CSV metrics và ảnh trước khi nhận xét số liệu.
2) Phân biệt rõ ảnh EDA, ảnh kết quả classification, clustering và luật kết hợp.
3) Nếu phát hiện thiếu artifact hoặc lệch số, chỉ rõ file nguồn, file cần sinh
   lại và lệnh chạy; không bịa số liệu.
4) Với báo cáo/PPT, dùng artifact trong reports/ và đảm bảo caption/bảng/ảnh
   cùng một lượt chạy. Không dùng ảnh cũ.
5) Nếu được yêu cầu chỉnh sửa, mô tả thay đổi, chạy test phù hợp và báo lại
   “Đạt/Chưa đạt” với bằng chứng.

Tình trạng Git: project đang ở nhánh recovery/telco-project-cleanup, có thay
đổi chưa commit và một backup ngoài project. Không dùng git reset --hard hoặc
xóa backup nếu chưa có xác nhận trực tiếp.

Bây giờ hãy xác nhận bạn đã hiểu bối cảnh theo đúng 3 phần: phân lớp, gom cụm
và luật kết hợp; sau đó chờ yêu cầu tiếp theo của tôi.
```
