# 🌿 Kế hoạch Thực hiện Project 1: Deep Learning - PlantVillage 

**Mục tiêu:** Xây dựng hệ thống phân loại bệnh trên lá cây sử dụng bộ dữ liệu PlantVillage. Phân công 4 thành viên theo quy tắc **"Ai code phần nào, viết report phần đó"**.
**Thời hạn:** 6 Ngày Sprint.

---

## 👥 Bảng Phân Công Nhiệm Vụ (Roles & Responsibilities)

### 🧑‍💻 Thành viên A: Data Master (Chịu trách nhiệm về Dữ liệu)
* **Phần Code (`data_prep.py` hoặc `.ipynb`):**
  - Tải dữ liệu từ Kaggle sử dụng `kagglehub`.
  - Khám phá dữ liệu (EDA): Hiển thị số lượng ảnh, số lượng class (bệnh), thống kê kích thước ảnh.
  - Phân chia tập dữ liệu: Chia thành 3 tập **Train (70%) / Validation (15%) / Test (15%)**. Đảm bảo thư mục lưu trữ rõ ràng để 3 thành viên khác dễ dàng sử dụng.
  - Tiền xử lý (Preprocessing): Resize ảnh (vd: 224x224), chuẩn hóa pixel (`1./255`).
  - Data Augmentation: Sử dụng `ImageDataGenerator` để thiết lập các phép xoay (rotation), lật (flip), zoom để tránh overfitting trên tập Train.
* **Phần Viết Báo cáo (Word):**
  - Viết **Phần I (Problem Description)**: Nêu mục tiêu nghiên cứu, input, output.
  - Viết **Phần II (Dataset Description)**: Kèm theo link download, thống kê số lượng ảnh, ví dụ trực quan về các loại lá.
  - Viết **Phần III (Data Preprocessing)**: Viết chi tiết về các bước làm sạch, chuẩn hóa, augment, và cách phân chia dữ liệu.

### 🧑‍💻 Thành viên B: Simple CNN Developer
* **Phần Code (`simple_cnn.py`):**
  - Tải tập dữ liệu đã xử lý từ Thành viên A.
  - Tự thiết kế một mô hình **Simple CNN**: Chỉ sử dụng các thành phần cơ bản (2-3 lớp `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`).
  - Huấn luyện mô hình, lưu lại file weight (`.h5` hoặc `.keras`).
  - Đánh giá trên tập Test chung: In ra Accuracy, Precision, Recall, F1-Score và vẽ đồ thị (Loss/Accuracy).
* **Phần Viết Báo cáo (Word):**
  - Hỗ trợ viết chung Lý thuyết **Phần 1: Convolutional Layer**.
  - Viết mục **Thiết kế Simple CNN** trong Phần III: Vẽ và giải thích kiến trúc mô hình của mình.
  - Điền kết quả thực nghiệm và nhận xét mô hình vào **Phần IV (Experimental Results)**.

### 🧑‍💻 Thành viên C: Complex CNN Developer
* **Phần Code (`complex_cnn.py`):**
  - Thiết kế một mô hình **Complex CNN**: Mô hình sâu hơn, có cấu trúc Block (như VGG-style block).
  - Áp dụng các kỹ thuật nâng cao: `BatchNormalization`, `Dropout` để mô hình hoạt động hiệu quả hơn Simple CNN.
  - Huấn luyện, đánh giá (Accuracy, Precision, Recall, F1-Score) và vẽ biểu đồ.
* **Phần Viết Báo cáo (Word):**
  - Hỗ trợ viết chung Lý thuyết **Phần 1: Pooling & Padding, Stride**.
  - Viết mục **Thiết kế Complex CNN** trong Phần III: Vẽ kiến trúc, giải thích lý do dùng BatchNormalization, Dropout.
  - Điền kết quả thực nghiệm và nhận xét vào **Phần IV (Experimental Results)**.

### 🧑‍💻 Thành viên D: Transfer Learning Expert
* **Phần Code (`transfer_learning.py`):**
  - Khai báo một Pre-trained model (VD: `MobileNetV2` hoặc `VGG16` từ Keras Applications).
  - Fine-tuning: Đóng băng (Freeze) các lớp gốc, tự xây dựng thêm một vài lớp `Dense` để phân loại các class của PlantVillage.
  - Huấn luyện (với Learning rate nhỏ), đánh giá và vẽ biểu đồ kết quả (dự kiến mô hình này sẽ cho kết quả rất cao).
* **Phần Viết Báo cáo (Word):**
  - Hỗ trợ viết chung Lý thuyết **Phần 1: Fully Connected Layer**.
  - Viết mục **Thiết kế Transfer Learning** trong Phần III: Giới thiệu mô hình pre-trained đã chọn, cách đóng băng layer, kiến trúc đoạn cuối.
  - Điền kết quả thực nghiệm và so sánh tổng quan vào **Phần IV (Experimental Results)**.

---

## 🗓 Lịch Trình Thực Thi 6 Ngày
* **Ngày 1:** Lập kho chứa GitHub, Thành viên A viết code Data Prep (Tải & chia tập), các thành viên khác clone repo và chia nhau viết Phần 1 báo cáo (Lý thuyết).
* **Ngày 2:** Bắt đầu dựng khung. Thành viên A viết xong Data Prep report. B, C, D tạo script riêng, thiết kế sơ đồ mô hình và viết cấu trúc vào báo cáo.
* **Ngày 3:** B, C, D ném mô hình lên Google Colab / Kaggle và tiến hành Training. A hỗ trợ fix lỗi pipeline data nếu có.
* **Ngày 4:** B, C, D tính toán Metrics (Accuracy, Precision, Recall, F1), xuất biểu đồ Loss/Accuracy và đẩy vào kho GitHub. Viết hoàn chỉnh báo cáo phần IV.
* **Ngày 5:** Hợp nhất các phần báo cáo. Lập bảng so sánh 3 mô hình. Cả 4 thành viên cùng thảo luận viết Phần V: Kết luận.
* **Ngày 6:** Review chéo bài (A đọc của B, B đọc C,...), chuẩn hóa format Word, thêm References, tạo file README.md cho GitHub và hoàn thành.

---
## 💡 Quy ước làm việc trên GitHub
1. Thư mục dữ liệu (vd: `dataset/`) không đẩy lên GitHub (nên thêm vào `.gitignore`). Thành viên A chỉ đẩy **code tạo dữ liệu**.
2. Khi code xong phần nào, luôn nhớ Commit với thông điệp rõ ràng, vd: `Thành viên A: Hoàn thành chia tập Train/Val/Test`.
3. Tránh sửa chung 1 file code, mỗi người code trên 1 file riêng biệt theo đúng tên đã phân công để không bao giờ bị conflict mã nguồn.
