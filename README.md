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

### 1. Cài đặt thư viện yêu cầu
Khuyến khích chạy mã nguồn này trên **Google Colab** hoặc **Kaggle Notebooks** để sử dụng GPU miễn phí. Nếu chạy trên máy cá nhân, hãy cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

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

## 📊 Kết quả thực nghiệm
Phần này sẽ được cập nhật sau ngày Training (Ngày 4).
Bao gồm:
- Bảng so sánh các chỉ số **Accuracy, Precision, Recall, F1-Score** giữa 4 mô hình.
- Biểu đồ **Loss/Accuracy** theo các epochs.
- Ma trận nhầm lẫn (Confusion Matrix).
