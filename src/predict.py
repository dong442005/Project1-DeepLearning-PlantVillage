# File đánh giá và dự đoán mô hình trên tập test (PlantVillage)
# Hỗ trợ đánh giá cả 4 mô hình hoặc dự đoán thử trên 1 ảnh cụ thể
import os
import sys
import csv

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font (cp1252 / charmap)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import resnet50, mobilenet_v2
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from model_loader import load_saved_model

# 1. Đường dẫn dữ liệu và thư mục lưu kết quả
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
OUTPUT_DIR = 'outputs/comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đường dẫn tới các file mô hình đã huấn luyện
MODEL_FILES = {
    'simple_cnn': 'models/simple_cnn_model.h5',
    'complex_cnn': 'models/complex_cnn_best.keras',
    'resnet50': 'models/resnet50_finetuned_model.h5',
    'mobilenetv2': 'models/mobilenetv2_finetuned_model.h5',
}

# Cấu hình kiểm tra:
# - Để IMAGE_PATH = None nếu muốn đánh giá toàn bộ tập test
# - Điền đường dẫn ảnh (ví dụ: 'data/test/Tomato___healthy/xxxx.jpg') để dự đoán 1 ảnh
IMAGE_PATH = None
TEST_STEPS = None  # None: chạy hết tập test; đặt số nhỏ (vd: 2) nếu muốn chạy thử nhanh


# 2. Hàm tiền xử lý ảnh cho từng mô hình (giống lúc train)
def preprocess_simple(img):
    mean = np.array([0.46658055, 0.48930454, 0.41043331])
    std = np.array([0.1992171, 0.17497431, 0.21742669])
    return (img / 255.0 - mean) / std

def preprocess_complex(img):
    mean = np.array([0.4666, 0.4893, 0.4104], dtype=np.float32)
    std = np.array([0.1992, 0.1750, 0.2174], dtype=np.float32)
    return (img / 255.0 - mean) / std

PREPROCESS_FUNCS = {
    'simple_cnn': preprocess_simple,
    'complex_cnn': preprocess_complex,
    'resnet50': resnet50.preprocess_input,
    'mobilenetv2': mobilenet_v2.preprocess_input,
}


# 3. Lấy danh sách tên nhãn lớp từ tập train
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print(f"Tổng số lớp: {len(class_names)}")


# 4. Thực hiện dự đoán
if IMAGE_PATH:
    # Trường hợp 1: Dự đoán nhãn cho 1 ảnh đơn lẻ
    print(f"\nDự đoán cho ảnh: {IMAGE_PATH}")
    img = load_img(IMAGE_PATH, target_size=(224, 224))
    img_array = img_to_array(img)

    for model_name, model_path in MODEL_FILES.items():
        if not os.path.exists(model_path):
            continue
        print(f"\n--- Model: {model_name} ---")
        model = load_saved_model(model_path)
        
        # Tiền xử lý và tạo batch
        x = PREPROCESS_FUNCS[model_name](img_array.copy())
        x = np.expand_dims(x, axis=0)

        # Dự báo nhãn
        y_prob = model.predict(x)
        pred_idx = np.argmax(y_prob, axis=1)[0]
        confidence = np.max(y_prob) * 100

        print(f"Nhãn dự đoán: {class_names[pred_idx]}")
        print(f"Độ tin cậy: {confidence:.2f}%")

else:
    # Trường hợp 2: Đánh giá và so sánh cả 4 mô hình trên tập test
    summary_results = []

    for model_name, model_path in MODEL_FILES.items():
        if not os.path.exists(model_path):
            print(f"Bỏ qua {model_name} vì chưa có file: {model_path}")
            continue

        print(f"\n==========================================")
        print(f"Đang đánh giá mô hình: {model_name}")
        print(f"==========================================")

        # Đọc dữ liệu tập test bằng ImageDataGenerator (tắt shuffle để giữ đúng thứ tự nhãn)
        test_datagen = ImageDataGenerator(preprocessing_function=PREPROCESS_FUNCS[model_name])
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(224, 224),
            batch_size=16,
            classes=class_names,
            class_mode='categorical',
            shuffle=False
        )

        # Load model đã huấn luyện
        model = load_saved_model(model_path)

        # Dự đoán nhãn trên tập test
        steps = min(TEST_STEPS, len(test_generator)) if TEST_STEPS else None
        y_prob = model.predict(test_generator, steps=steps)
        y_pred = np.argmax(y_prob, axis=1)
        y_test = test_generator.classes[:len(y_pred)]

        # Đánh giá chất lượng dự báo của model
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )

        print(f"Số lượng mẫu test: {len(y_test)}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        summary_results.append([model_name, len(y_test), acc, precision, recall, f1])

        # Lưu báo cáo chi tiết classification_report
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        report_file = os.path.join(OUTPUT_DIR, f"{model_name}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # Lưu chi tiết dự đoán từng ảnh ra file CSV
        csv_file = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'true_label', 'predicted_label'])
            for path, actual, pred in zip(test_generator.filenames, y_test, y_pred):
                writer.writerow([path, class_names[actual], class_names[pred]])

    # In bảng tổng kết so sánh các model
    if summary_results:
        print("\n" + "=" * 65)
        print("BẢNG TỔNG KẾT SO SÁNH CÁC MÔ HÌNH")
        print("=" * 65)
        print(f"{'Model':<18} {'Images':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 65)
        for name, n_samples, acc, prec, rec, f1_val in summary_results:
            print(f"{name:<18} {n_samples:<8} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1_val:<12.4f}")

        # Ghi bảng so sánh ra CSV
        comparison_csv = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
        with open(comparison_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'images', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1'])
            writer.writerows(summary_results)
        print(f"\nĐã lưu kết quả so sánh vào {comparison_csv}")
