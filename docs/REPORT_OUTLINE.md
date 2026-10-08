# 📑 MỤC LỤC DỰ KIẾN KÈM PHÂN CÔNG BÁO CÁO (Word)

Đây là khung cấu trúc báo cáo chuẩn dựa trên chính xác yêu cầu từ file `Major_Assignment_Report_Project 1_2026.docx` của trường bạn. Tên của người phụ trách được tag trực tiếp vào từng mục để mọi người không bị dẫm chân lên nhau.

---

**PHẦN 1: CƠ SỞ LÝ THUYẾT VỀ MẠNG NƠ-RON TÍCH CHẬP (CNN)**
- 1.1. Khái niệm và Kiến trúc chung của mạng CNN **(Thành viên B)**
- 1.2. Phép toán Tích chập (Convolution Operation) **(Thành viên B)**
- 1.3. Khái niệm Padding và Stride **(Thành viên C)**
- 1.4. Lớp Gộp (Pooling Layer - Max Pooling & Average Pooling) **(Thành viên C)**
- 1.5. Lớp Kết nối Đầy đủ (Fully Connected Layer) **(Thành viên D)**
- 1.6. Kỹ thuật Transfer Learning (Học chuyển giao) **(Thành viên D)**

**PHẦN 2: THỰC HÀNH ỨNG DỤNG**

**I. MÔ TẢ BÀI TOÁN (PROBLEM DESCRIPTION) -> (Thành viên A)**
- 1. Tiêu đề bài toán (Phân loại bệnh trên lá cây với PlantVillage)
- 2. Mục tiêu nghiên cứu
- 3. Đầu vào (Input) và Đầu ra (Output)
- 4. Tóm tắt các tác vụ đã thực hiện

**II. MÔ TẢ TẬP DỮ LIỆU (DATASET DESCRIPTION) -> (Thành viên A)**
- 1. Đường dẫn tải dữ liệu (Link Kaggle)
- 2. Đặc điểm tập dữ liệu (Số lượng ảnh, loại bệnh/loại lá, kích thước ảnh)

**III. THIẾT KẾ CÁC MÔ HÌNH CNN (CNN MODEL DESIGN)**
- 1. Tiền xử lý dữ liệu: Làm sạch, chuẩn hóa, tăng cường dữ liệu **(Thành viên A)**
- 2. Phân chia tập dữ liệu: Train/Val/Test (Kèm số lượng mẫu cụ thể mỗi tập) **(Thành viên A)**
- 3. Cấu hình huấn luyện: Siêu tham số và Môi trường (Colab GPU) **(Cả nhóm / Leader tổng hợp)**
- 4. Các thang đo đánh giá: Trình bày công thức Toán học **(Cả nhóm / Leader tổng hợp)**
- 5. Mô hình 1: Thiết kế mạng Simple CNN **(Thành viên B)**
- 6. Mô hình 2: Thiết kế mạng Complex CNN (Sử dụng Blocks, BatchNorm, Dropout) **(Thành viên C)**
- 7. Mô hình 3: Thiết kế mạng Transfer Learning với MobileNetV2 **(Thành viên D)**
- 8. Mô hình 4: Thiết kế mạng Transfer Learning với ResNet50 **(Thành viên A)**

**IV. KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)**
*(Mỗi mô hình bên dưới trình bày Biểu đồ Loss/Accuracy và Ma trận nhầm lẫn)*
- 1. Kết quả của Simple CNN **(Thành viên B)**
- 2. Kết quả của Complex CNN **(Thành viên C)**
- 3. Kết quả của MobileNetV2 **(Thành viên D)**
- 4. Kết quả của ResNet50 **(Thành viên A)**
- 5. Bảng tổng hợp Đánh giá (So sánh 4 mô hình qua Accuracy, Precision, Recall, F1-Score) **(Leader tổng hợp)**
- 6. Thảo luận và Phân tích lỗi (Error Analysis) dựa trên Ma trận nhầm lẫn **(Cả 4 thành viên cùng thảo luận)**

**V. KẾT LUẬN (CONCLUSION) -> (Cả 4 thành viên cùng chắp bút)**
- 1. Tóm tắt các kết quả chính mà nhóm đạt được
- 2. Lựa chọn mô hình tối ưu nhất cho thực tế 
- 3. Hướng phát triển (Future Work)

**TÀI LIỆU THAM KHẢO (REFERENCES)**
*(Ai viết lý thuyết hoặc code phần nào tự chèn link tham khảo phần đó)*
