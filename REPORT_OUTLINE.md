# Outline chi tiết và phân công nhiệm vụ (Báo cáo Bài tập lớn Deep Learning)

**Đề tài:** Plant Disease Classification using Custom CNNs and Transfer Learning

---

## PART 1: THEORETICAL BACKGROUND OF CONVOLUTIONAL NEURAL NETWORKS (CNNs)
1.1. Concept and General Architecture of CNNs – Phạm Vân Thư
1.2. Convolution Operation – Phạm Vân Thư
1.3. Padding and Stride – Nguyễn Thị Thu Trang
1.4. Pooling Layer (Max Pooling & Average Pooling) – Nguyễn Thị Thu Trang
1.5. Fully Connected Layer – Nguyễn Việt Hằng
1.6. Transfer Learning – Nguyễn Việt Hằng

---

## PART 2: PRACTICAL APPLICATION

### I. PROBLEM DESCRIPTION – Nguyễn Phương Đông
- Problem Title: Plant Disease Classification Using the PlantVillage Dataset
- Research Objectives
- Input and Output of the Problem
- Summary of Tasks Performed

### II. DATASET DESCRIPTION – Nguyễn Phương Đông
- Dataset Source and Acquisition (Sử dụng PlantVillage dataset từ Kaggle qua kagglehub)
- Dataset Characteristics (54,305 ảnh RGB, kích thước chuẩn 224x224x3, 14 loại cây trồng, 38 lớp, vấn đề mất cân bằng dữ liệu)
- Descriptive Statistics of the Dataset

### III. CNN MODEL DESIGN
- 1. Data Preprocessing and Dataset Splitting – Nguyễn Phương Đông (Làm sạch, chuẩn hóa, tăng cường dữ liệu và chia tập Train/Validation/Test)
- 2. Training Configuration – Leader tổng hợp (Môi trường huấn luyện, số epochs, batch size, optimizer, learning rate, hàm loss...)
- 3. Evaluation Metrics – Leader tổng hợp (Accuracy, Precision, Recall, F1-Score)
- 4. Model 1: Simple CNN – Phạm Vân Thư (Kiến trúc mô hình, các lớp tích chập, pooling, fully connected)
- 5. Model 2: Complex CNN – Nguyễn Thị Thu Trang (CNN blocks, các lớp tích chập, pooling, fully connected, Batch Normalization/Dropout)
- 6. Model 3: Transfer Learning with MobileNetV2 – Nguyễn Việt Hằng (Kiến trúc, phương pháp Transfer Learning/Fine-Tuning, cấu hình huấn luyện)
- 7. Model 4: Transfer Learning with ResNet50 – Nguyễn Phương Đông (Kiến trúc, phương pháp Transfer Learning/Fine-Tuning, cấu hình huấn luyện)

### IV. EXPERIMENTAL RESULTS
*(Báo cáo hiệu suất trên tập test, bảng các chỉ số Accuracy, Precision, Recall, F1-Score và biểu đồ Loss/Accuracy)*
- Results of Simple CNN – Phạm Vân Thư
- Results of Complex CNN – Nguyễn Thị Thu Trang
- Results of MobileNetV2 – Nguyễn Việt Hằng
- Results of ResNet50 – Nguyễn Phương Đông
- Overall Comparison of the Four Models – Leader tổng hợp (Bảng so sánh tổng quan cả 4 mô hình)

### V. CONCLUSION – Toàn bộ 4 thành viên
- Summary of the Main Results
- Best-Performing Model
- Future Work

### REFERENCES
*Mỗi thành viên chịu trách nhiệm bổ sung tài liệu tham khảo và link nguồn cho phần lý thuyết hoặc code thuộc phần mình phụ trách.*
