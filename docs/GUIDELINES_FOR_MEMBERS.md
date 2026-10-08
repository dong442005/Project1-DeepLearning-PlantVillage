# 🎯 HƯỚNG DẪN CHI TIẾT DÀNH CHO CÁC THÀNH VIÊN (Kèm Định hướng sử dụng AI)

Tài liệu này được biên soạn để định hướng chi tiết cho từng thành viên. 
**LƯU Ý QUAN TRỌNG:** Nếu các bạn sử dụng ChatGPT, Claude hay Copilot để hỗ trợ viết code/viết báo cáo, **HÃY COPY Y NGUYÊN NỘI DUNG PHẦN CỦA BẠN TRONG FILE NÀY DÁN VÀO AI** để AI sinh ra code/bài viết đi đúng hướng môn học, tránh việc AI sinh ra code PyTorch hoặc code quá phức tạp, lệch khỏi yêu cầu của giảng viên!

---

## 🛑 QUY TẮC CHUNG CHO CẢ NHÓM (Dán cái này vào AI trước tiên)
> "Act as a Deep Learning student. All code must be written in **Python using TensorFlow/Keras**. Use `Sequential` API or Functional API. When writing the report, explain concepts simply using standard formulas like Output size = `(H - h + 1)x(W - w + 1)`. Ensure image labels are one-hot encoded and loss is `categorical_crossentropy`. Plot graphs using `matplotlib` with clear grids and legends."

---

## 🧑‍💻 THÀNH VIÊN A: Data Master & ResNet50 
**1. Trách nhiệm Code:**
- **Data Pipeline:** Đã hoàn thiện (file `data_prep.py`).
- **Mô hình ResNet50:** Bạn phải code file `resnet_model.py`. Sử dụng `ResNet50` từ `keras.applications`.
- **Định hướng AI (Prompt cho AI):** 
  > "Viết một script Python sử dụng Keras để build mô hình Transfer Learning. Hãy load `ResNet50` (weights='imagenet', include_top=False). Đóng băng (freeze) toàn bộ base model. Thêm `GlobalAveragePooling2D()` và lớp `Dense(num_classes, activation='softmax')`. Code hàm compile với thuật toán `Adam` và `categorical_crossentropy`. Sau khi train xong phần đầu, mở băng (unfreeze) 10 lớp cuối của ResNet50 và train với learning rate cực nhỏ (fine-tuning). Lưu model ra file `.h5` và dùng matplotlib in ra lịch sử loss/accuracy."

**2. Định hướng Viết Báo cáo:**
- Viết Phần I (Mô tả bài toán) và Phần II (Dữ liệu).
- Ở Phần III (Tiền xử lý): Nhấn mạnh việc chuyển nhãn sang **One-Hot Encoding** (đúng như lý thuyết cô dạy) và việc resize ảnh về 224x224 để phù hợp với chuẩn đầu vào của mạng ResNet.

---

## 🧑‍💻 THÀNH VIÊN B: Simple CNN (Lấy cảm hứng từ LeNet-5)
**1. Trách nhiệm Code:**
- Code file `simple_cnn.py`. Không dùng mạng có sẵn, phải tự thiết kế một mạng chập nhỏ và cơ bản nhất.
- **Định hướng AI (Prompt cho AI):** 
  > "Viết một script Python Keras Sequential API tạo một mô hình mạng CNN lấy cảm hứng từ mạng **LeNet-5** nhưng được điều chỉnh cho ảnh RGB kích thước 224x224. Mô hình gồm 2 lớp `Conv2D` (kernel 5x5 hoặc 3x3), sau mỗi lớp Conv là một lớp `AveragePooling2D` hoặc `MaxPooling2D`. Tiếp theo là `Flatten` và 2 lớp `Dense` (kích thước giảm dần, dùng activation='relu'). Cuối cùng là `Dense` cho multi-class classification với 'softmax'. In ra model.summary(). Viết code train mô hình này và lưu kết quả."

**2. Định hướng Viết Báo cáo:**
- Ở Phần I (Lý thuyết): Mở file slide `extract.txt` của môn học, chép lại lý thuyết về Convolution (Cross-correlation), Padding, Stride, Pooling.
- Ở Phần III (Thiết kế): Chèn ảnh cấu trúc mô hình. **Câu ăn điểm:** *"Dựa vào bài giảng Chương 5 về mạng LeNet, nhóm đã áp dụng và mở rộng tư duy của LeNet-5 để thiết kế mạng Simple CNN xử lý ảnh màu kích thước lớn hơn, dùng làm mô hình cơ sở đánh giá độ khó của bài toán."*

---

## 🧑‍💻 THÀNH VIÊN C: Complex CNN (Mạng tích chập sâu)
**1. Trách nhiệm Code:**
- Code file `complex_cnn.py`. Xây dựng mô hình tự thiết kế (từ con số 0) nhưng phải sâu và hiện đại hơn mạng của Thành viên B.
- **Định hướng AI (Prompt cho AI):** 
  > "Viết một script Keras tạo một mô hình Custom CNN sâu theo phong cách VGG (VGG-style blocks) cho ảnh 224x224. Tạo khoảng 3 đến 4 khối (blocks). Mỗi khối gồm: 2 lớp `Conv2D` (kernel 3x3, padding='same', activation='relu') -> 1 lớp `BatchNormalization` -> 1 lớp `MaxPooling2D`. Sau khi qua các khối, thêm `Flatten`, 1 lớp `Dense` cỡ 512, thêm `Dropout(0.5)` để chống overfitting. Cuối cùng là output layer. Code phần evaluate tính Accuracy, Precision, Recall, F1-score dùng thư viện sklearn."

**2. Định hướng Viết Báo cáo:**
- Giải thích rõ tại sao mô hình của Thành viên B (Simple) dễ bị Overfitting, từ đó dẫn dắt sang việc mô hình Complex này phải áp dụng **Batch Normalization** (để hội tụ nhanh) và **Dropout** (để ép các nơ-ron học đặc trưng độc lập). Chú ý tham khảo bài giảng CIFAR-10 của cô giáo vì mô hình này tương tự bài đó nhưng quy mô lớn hơn.

---

## 🧑‍💻 THÀNH VIÊN D: Transfer Learning (MobileNetV2)
**1. Trách nhiệm Code:**
- Code file `transfer_learning.py`. Xây dựng mạng dựa trên **MobileNetV2** (đại diện cho mô hình hiện đại, trọng lượng nhẹ - nhẹ hơn nhiều so với ResNet50 của A).
- **Định hướng AI (Prompt cho AI):** 
  > "Viết script Keras để transfer learning bằng mô hình `MobileNetV2` (weights='imagenet', include_top=False). Mục tiêu là tạo ra mô hình nhẹ, chạy nhanh nhưng độ chính xác cao. Đóng băng các lớp gốc, thiết kế phần Fully Connected đơn giản ở cuối. Viết code train mô hình. Viết hàm vẽ Ma trận nhầm lẫn (Confusion Matrix) dùng thư viện seaborn và matplotlib để đánh giá xem mô hình hay nhầm lẫn loại bệnh nào nhất."

**2. Định hướng Viết Báo cáo:**
- Viết lý thuyết về **Fully Connected Layer** và **Transfer Learning**.
- Ở Phần IV (Đánh giá kết quả): Hãy so sánh trực tiếp kết quả của MobileNetV2 (nhẹ, tính toán nhanh) với ResNet50 (nặng, phức tạp) của Thành viên A để xem trên dữ liệu PlantVillage, liệu mô hình phức tạp hơn có thực sự tốt hơn không. Sự phân tích này sẽ làm báo cáo cực kỳ có chiều sâu.
