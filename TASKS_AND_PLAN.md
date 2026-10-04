# 🌿 Kế hoạch Thực hiện Project 1: Deep Learning - PlantVillage 

**Mục tiêu:** Xây dựng hệ thống phân loại bệnh trên lá cây sử dụng bộ dữ liệu PlantVillage. Phân công 4 thành viên theo quy tắc **"Ai code phần nào, viết report phần đó"**.
**Thời hạn:** 6 Ngày Sprint.

---

## 👥 Bảng Phân Công Nhiệm Vụ (Roles & Responsibilities)

### 🧑‍💻 Thành viên A: Data Master & Transfer Learning (ResNet50)
* **Phần Code (`data_prep.py` và `resnet_model.py`):**
  - Đảm nhiệm toàn bộ quy trình chuẩn bị dữ liệu chung: Tải tập dữ liệu, chia tập Train (70%) / Val (15%) / Test (15%) và thiết lập Data Augmentation để 3 thành viên còn lại dùng chung.
  - Tự thiết kế và huấn luyện thêm một mô hình Transfer Learning "hạng nặng": **ResNet50** (để so sánh với mô hình MobileNetV2 của Thành viên D).
  - Đánh giá mô hình của mình trên tập Test.
* **Phần Viết Báo cáo (Word):**
  - Viết **Phần I (Problem Description)**, **Phần II (Dataset Description)** và **Phần III (Data Preprocessing)**.
  - Viết mục thiết kế cấu trúc mạng **ResNet50** trong Phần III và điền kết quả vào Phần IV.

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

## 💡 Quy tắc làm việc Nhóm bằng Git (Git Workflow)
Để tránh việc code đè lên nhau gây mất dữ liệu (Conflict), nhóm TUYỆT ĐỐI tuân thủ quy trình tạo nhánh (Branching) như sau:

**1. Không code trực tiếp trên nhánh `main`:** Nhánh `main` chỉ dùng để lưu code đã hoàn chỉnh và chạy ngon lành.
**2. Mỗi người tạo một nhánh riêng (Branch) để làm việc.** 
- Cú pháp tạo nhánh và chuyển sang nhánh đó: `git checkout -b <tên-nhánh>`
- Ví dụ cách đặt tên nhánh: 
  + B: `git checkout -b feature/simple-cnn`
  + C: `git checkout -b feature/complex-cnn`
  + D: `git checkout -b feature/transfer-learning`

**3. Quy trình code hàng ngày của từng cá nhân:**
- **Bước 1:** Lấy code mới nhất từ nhánh main về máy: `git pull origin main`
- **Bước 2:** Chuyển sang nhánh của mình: `git checkout feature/<tên-nhánh-của-bạn>`
- **Bước 3:** Mở VSCode/Google Colab và bắt đầu code vào file được phân công.
- **Bước 4:** Code xong thì đẩy lên nhánh của TỰ MÌNH:
  ```bash
  git add .
  git commit -m "Tên bạn: Nội dung vừa code xong"
  git push origin feature/<tên-nhánh-của-bạn>
  ```
- **Bước 5:** Lên trang web GitHub, bấm nút **"Compare & pull request"** để yêu cầu gộp code vào `main`. Leader sẽ duyệt.

**4. Quy tắc vàng:**
- Thư mục ảnh (`dataset/`) KHÔNG ĐƯỢC đẩy lên Git (đã chặn ở `.gitignore`).
- Tránh sửa chung 1 file code. Ai phụ trách file nào thì chỉ gõ vào file đó. Mọi thứ sẽ không bao giờ bị conflict!
