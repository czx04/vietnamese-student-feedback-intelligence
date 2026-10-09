# Vietnamese Student Feedback Intelligence

Hệ thống NLP phân tích phản hồi sinh viên bằng tiếng Việt, được xây dựng từ quá trình huấn luyện mô hình đến API và giao diện sử dụng hoàn chỉnh.

## Bài toán

Dự án xử lý đồng thời hai tác vụ trên bộ dữ liệu UIT-VSFC:

- Phân loại cảm xúc: tích cực, trung lập hoặc tiêu cực.
- Phân loại chủ đề: giảng viên, chương trình đào tạo, cơ sở vật chất hoặc chủ đề khác.

## Thành phần chính

- Pipeline tiền xử lý, huấn luyện và đánh giá mô hình.
- Baseline TF-IDF kết hợp Linear SVM.
- Fine-tuning PhoBERT và so sánh với baseline.
- REST API bằng FastAPI.
- Lưu trữ kết quả phân tích bằng PostgreSQL.
- Giao diện React một trang để nhập và phân tích phản hồi.
- Đóng gói toàn bộ hệ thống bằng Docker Compose.

## Kết quả PhoBERT trên tập test

- Cảm xúc: Accuracy **94,35%**, Macro-F1 **83,61%**.
- Chủ đề: Accuracy **89,26%**, Macro-F1 **79,85%**.


## Kiến trúc tổng quát

`React → FastAPI → mô hình NLP → PostgreSQL`
