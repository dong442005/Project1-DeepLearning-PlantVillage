# Plant Disease Classification using Custom CNNs and Transfer Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green?style=flat)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

Group 3, Project 1 Deep Learning: Phân loại bệnh trên lá cây dựa trên bộ dữ liệu PlantVillage thành **38 lớp (bệnh và lá khỏe mạnh)**. Mục dích là so sánh hiệu năng giữa 2 mô hình CNN tự xây dựng (train từ đầu) và 2 mô hình Transfer Learning (pretrained từ ImageNet):


- **Simple CNN:** Baseline 3 khối tích chập lấy cảm hứng từ kiến trúc LeNet-5.
- **Complex CNN:** 4 khối tích chập kết hợp Batch Normalization, Max Pooling và Dropout.
- **ResNet50:** Transfer learning với kiến trúc mạng phần dư (Residual Network).
- **MobileNetV2:** Transfer learning với kiến trúc mạng nhẹ tối ưu cho thiết bị di động.

---

## 1. Dataset

Dữ liệu ảnh được tải từ [PlantVillage Kaggle dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) ([PlantVillage original repo](https://github.com/spMohanty/PlantVillage-Dataset)).

- **Phân chia tập dữ liệu:** ~70% Train, 15% Validation, 15% Test (chia đều theo từng lớp với `random_state=42`).
- **Kích thước đầu vào:** Ảnh màu RGB 224 × 224.
- **Tiền xử lý (Preprocessing):** Chuẩn hóa Z-score cho Simple CNN & Complex CNN; dùng `preprocess_input` riêng cho ResNet50 và MobileNetV2.
- **Tăng cường dữ liệu (Augmentation):** Xoay góc (rotation), dịch chuyển (shifts), zoom, lật ngang (horizontal flip); mô hình ResNet50 có thêm biến dạng (shear).

> **Lưu ý:** Thư mục dữ liệu `data/` được loại trừ khỏi Git. Để đảm bảo tính nhất quán khi đánh giá checkpoint, nên giữ nguyên phân vùng dữ liệu sau khi chạy script chia tập.

---

## 2. Cấu trúc thư mục

```text
Project1-DeepLearning-PlantVillage/
├── src/
│   ├── data_prep.py           # Tải và chia tập dữ liệu Train / Val / Test
│   ├── simple_cnn.py          # Huấn luyện và đánh giá Simple CNN
│   ├── complex_cnn.py         # Huấn luyện Complex CNN
│   ├── resnet_model.py        # Huấn luyện và đánh giá ResNet50
│   ├── transfer_learning.py   # Huấn luyện và đánh giá MobileNetV2
│   ├── predict.py             # Đánh giá các model hoặc dự đoán thử 1 ảnh
│   └── model_loader.py        # Hỗ trợ nạp model tương thích giữa các phiên bản Keras
├── notebooks/                 # Khảo sát dữ liệu và thử nghiệm mô hình
├── outputs/                   # Đồ thị huấn luyện, ma trận nhầm lẫn và báo cáo
├── data/                      # Dữ liệu train/, val/, test/ (tạo khi chạy data_prep.py)
├── models/                    # File trọng số mô hình đã huấn luyện
├── requirements.txt           # Danh sách thư viện cần thiết
└── README.md
```

---

## 3. Cài đặt & Hướng dẫn chạy

Khuyến nghị sử dụng **Python 3.11** và thực thi lệnh từ thư mục gốc của repository.

### 3.1 Cài đặt môi trường

**Trên macOS / Linux:**
```bash
git clone https://github.com/dong442005/Project1-DeepLearning-PlantVillage.git
cd Project1-DeepLearning-PlantVillage
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

**Trên Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(Môi trường huấn luyện sử dụng TensorFlow và Keras 2.15.0).*

---

### 3.2 Chuẩn bị dữ liệu

```bash
python src/data_prep.py
```
Script sẽ tự động chuẩn bị thư mục `data/train/`, `data/val/` và `data/test/`.  
*Lưu ý: Nếu chạy lại script, thư mục `data/` hiện tại sẽ được khởi tạo lại.*

---

### 3.3 Huấn luyện mô hình

Chạy từng script tương ứng trong thư mục `src/`:

```bash
python src/simple_cnn.py
python src/complex_cnn.py
python src/resnet_model.py
python src/transfer_learning.py
```

- Các mô hình đều dùng `batch_size = 32`.
- **Simple CNN:** Mặc định 20 epoch (có thể chỉnh bằng `--epochs 30` hoặc `--smoke-test`).
- **Complex CNN:** Huấn luyện tối đa 30 epoch với Early Stopping (patience = 5). File model lưu tại `models/complex_cnn_best.keras`.
- **ResNet50 & MobileNetV2:** Huấn luyện 2 giai đoạn (10 epoch đầu đóng băng backbone + 10 epoch sau fine-tuning).
- Kết quả và biểu đồ sau khi train được lưu tự động trong thư mục `outputs/`.

---

### 3.4 Đánh giá và Dự đoán (Inference)

Nếu muốn chạy trực tiếp file checkpoint đã huấn luyện mà không cần train lại:

1. **Tải trọng số mô hình:** Tải 4 file model và đặt vào thư mục `models/`:

| Mô hình | Tên file | Link tải Google Drive |
| :--- | :--- | :--- |
| **Simple CNN** | `simple_cnn_model.h5` | [Tải về](https://drive.google.com/file/d/1Sm-wobQFy9BJxGX1k-ymcj9AwLZQYVGB/view?usp=drive_link) |
| **Complex CNN** | `complex_cnn_best.keras` | [Tải về](https://drive.google.com/file/d/1u4jGbdPBOSjgAoLPtIXldHX1aogQ54NW/view?usp=drive_link) |
| **ResNet50** | `resnet50_finetuned_model.h5` | [Tải về](https://drive.google.com/file/d/1UMcnfKdfKLHgxe-mLu5CJYCGoO7_8CiT/view?usp=drive_link) |
| **MobileNetV2** | `mobilenetv2_finetuned_model.h5` | [Tải về](https://drive.google.com/file/d/1cm6hZZtg2CG-PF4kxbcpPzBstv1DrXoi/view?usp=drive_link) |

2. **Chạy đánh giá:**
```bash
python src/predict.py
```

- Mặc định (`IMAGE_PATH = None`): script sẽ đánh giá cả 4 mô hình trên toàn bộ tập test và xuất bảng so sánh chi tiết ra `outputs/comparison/`.
- Để dự đoán thử trên một ảnh cụ thể: mở `src/predict.py` và điền đường dẫn ảnh vào biến `IMAGE_PATH` (ví dụ: `'data/test/Tomato___healthy/xxxx.jpg'`).

---

## 4. Kết quả thực nghiệm

Kết quả dưới đây được đánh giá đồng nhất trên cùng tập kiểm thử **8,162 ảnh test** thông qua pipeline `src/predict.py`:

| Mô hình | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **Simple CNN** | 95.03% | 92.98% | 94.92% | 93.80% |
| **Complex CNN** | 95.72% | 94.15% | 94.23% | 94.03% |
| **MobileNetV2** | 95.07% | 94.46% | 95.18% | 94.72% |
| **ResNet50** | **98.76%** | **98.79%** | **98.76%** | **98.76%** |

### Nhận xét:
- **ResNet50** đạt kết quả cao nhất ở toàn bộ các độ đo (Accuracy đạt 98.76%)
- **Complex CNN** đạt độ chính xác 95.72%, nhỉnh hơn một chút so với **Simple CNN** (95.03%)
- **MobileNetV2** đạt 95.07% và F1-score 94.72%

---

## 5. Phân công nhóm

| Thành viên | Nhiệm vụ thực hiện |
| :--- | :--- |
| **Phương Đông** | Chuẩn bị dữ liệu & Huấn luyện mô hình ResNet50 |
| **Vân Thư** | Xây dựng và đánh giá mô hình Simple CNN |
| **Thu Trang** | Xây dựng và đánh giá mô hình Complex CNN |
| **Việt Hằng** | Xây dựng và đánh giá mô hình MobileNetV2 |
