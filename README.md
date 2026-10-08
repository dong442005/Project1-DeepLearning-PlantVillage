# 🌿 Nhận Diện Bệnh Qua Lá Cây (PlantVillage Disease Classification)

Đây là mã nguồn cho **Project 1 - Môn Deep Learning**.

## 📌 Giới thiệu dự án
Dự án nhằm mục đích phân loại bệnh trên lá cây sử dụng bộ dữ liệu hình ảnh **PlantVillage**. Chúng tôi sẽ tiến hành tiền xử lý, tăng cường dữ liệu và thiết kế **4 kiến trúc mạng nơ-ron tích chập (CNN)** khác nhau để giải quyết bài toán đa phân loại (Multi-class Classification) này.

**Bộ dữ liệu:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 👥 Đội ngũ thực hiện & Kiến trúc Mô hình
Dự án được thực hiện bởi nhóm 4 người, phân chia kỹ thuật thành 4 khối lượng công việc như sau:
- **Thành viên A (Data Pipeline & ResNet50):** Xử lý toàn bộ khâu chuẩn bị dữ liệu (Data Augmentation, Pipeline) và tự huấn luyện thêm mô hình Transfer Learning ResNet50 (mạng sâu, tính học thuật).
- **Thành viên B (Simple CNN):** Tự thiết kế và huấn luyện mô hình mạng tích chập cơ sở (Baseline) lấy cảm hứng từ kiến trúc kinh điển LeNet-5.
- **Thành viên C (Complex CNN):** Thiết kế mạng tích chập chuyên sâu tự code (sử dụng Blocks, Batch Normalization, Dropout để chống Overfitting).
- **Thành viên D (Transfer Learning & MobileNetV2):** Tinh chỉnh (Fine-tuning) mạng MobileNetV2 đã được huấn luyện trước (mô hình nhẹ, tính ứng dụng thực tiễn cao).

🔗 **Tài liệu nội bộ cho Nhóm:**
- [Bảng phân công chi tiết (TASKS & PLAN)](TASKS_AND_PLAN.md)
- [Hướng dẫn AI & Định hướng Logic cho từng thành viên](GUIDELINES_FOR_MEMBERS.md)
- [Mục lục & Phân công Báo cáo Word](REPORT_OUTLINE.md)

## 📂 Cấu trúc thư mục (Folder Structure)
```text
Project1-DeepLearning-PlantVillage/
├── data/                        # (Bị ẩn bởi .gitignore) Chứa ảnh gốc chia theo train/val/test
├── notebooks/                   # Chứa các file jupyter notebook (.ipynb) khám phá dữ liệu
├── results/                     # Nơi lưu trữ biểu đồ và ảnh ma trận kết quả sau khi train
├── src/                         # Chứa toàn bộ source code Python
│   ├── data_prep.py             # (Thành viên A) Script tải và chia tập dữ liệu
│   ├── resnet_model.py          # (Thành viên A) Script train mô hình ResNet50
│   ├── simple_cnn.py            # (Thành viên B) Script train mô hình cơ sở LeNet-5
│   ├── complex_cnn.py           # (Thành viên C) Script train mô hình CNN sâu
│   └── transfer_learning.py     # (Thành viên D) Script train mô hình MobileNetV2
├── .gitignore                   # Chặn các file rác, file dataset nặng
├── README.md                    # Lời giới thiệu dự án
├── TASKS_AND_PLAN.md            # Bảng phân công nhiệm vụ và lịch trình
├── GUIDELINES_FOR_MEMBERS.md    # Hướng dẫn chi tiết & Prompt AI cho từng người
└── REPORT_OUTLINE.md            # Dàn ý mục lục và phân công viết báo cáo Word
```

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường và Thư viện
Nếu chạy trên máy cá nhân, nhóm TUYỆT ĐỐI thống nhất tạo Môi trường ảo (Virtual Environment) để không bị xung đột phiên bản như sau:

**Dùng Conda (Nhanh và tiện nhất):**
```bash
conda create -n dl_env python=3.10
conda activate dl_env
pip install -r requirements.txt
```

*(Hoặc dùng `python -m venv venv` nếu bạn không xài Conda).*

### 2. Tải và phân chia Dữ liệu
Để tự động tải dữ liệu và chia thành các thư mục con `train/`, `val/`, và `test/`, hãy chạy lệnh:
```bash
python src/data_prep.py
```
> **Chú ý:** Quá trình tải sẽ mất khoảng vài phút tùy thuộc vào mạng của bạn. Thư mục `data/` (nơi chứa ảnh) đã được cấu hình ẩn trong `.gitignore` để tránh đẩy dữ liệu khổng lồ lên Git.

### 3. Huấn luyện các Mô hình
Sau khi đã có thư mục `data/`, hãy chạy từng file code sau để huấn luyện 4 loại mô hình tương ứng:
- `python src/simple_cnn.py` *(Đang xây dựng...)*
- `python src/complex_cnn.py` *(Đang xây dựng...)*
- `python src/transfer_learning.py` *(Đang xây dựng - MobileNetV2)*
- `python src/resnet_model.py` *(Đang xây dựng - ResNet50)*

## 📊 Kết quả thực nghiệm và Mô hình (Models & Results)
Dưới đây là liên kết tải các mô hình (đã được lưu dưới dạng `.h5`) sau khi huấn luyện xong. Do kích thước file quá lớn (vượt giới hạn 100MB của GitHub), nhóm lưu trữ chúng trên Google Drive:

- 🧠 **ResNet50 Model (Thành viên A - 223MB):** [Tải về tại đây](https://drive.google.com/file/d/1DooI4k3YiHRk8JXDBMMcKT2jk37YSW_T/view?usp=drive_link)
- *(Các mô hình của Thành viên B, C, D sẽ được cập nhật sau)*

Phần đánh giá chi tiết (Bảng so sánh Accuracy, F1-Score, Biểu đồ) sẽ được trình bày cụ thể trong Báo cáo Word cuối kỳ của nhóm.
