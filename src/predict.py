import csv
import gc
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2, resnet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from model_loader import load_saved_model

# Đường dẫn dữ liệu và các model đã train
ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / 'data/test'
TRAIN_DIR = ROOT / 'data/train'
OUTPUT_DIR = ROOT / 'outputs/comparison'
MODEL_FILES = {
    'simple_cnn': ROOT / 'models/simple_cnn_model.h5',
    'complex_cnn': ROOT / 'models/complex_cnn_best.keras',
    'resnet50': ROOT / 'models/resnet50_finetuned_model.h5',
    'mobilenetv2': ROOT / 'models/mobilenetv2_finetuned_model.h5',
}

IMAGE_PATH = None  # Đổi thành đường dẫn ảnh nếu muốn predict một ảnh
TEST_STEPS = None  # None: toàn bộ test; 2: chạy thử 2 batch đầu


# Chuẩn hóa ảnh giống lúc train từng model
def normalize_simple(image):
    mean = np.array([0.46658055, 0.48930454, 0.41043331])
    std = np.array([0.1992171, 0.17497431, 0.21742669])
    return (image / 255.0 - mean) / std


def normalize_complex(image):
    mean = np.array([0.4666, 0.4893, 0.4104], dtype=np.float32)
    std = np.array([0.1992, 0.1750, 0.2174], dtype=np.float32)
    return (image / 255.0 - mean) / std


PREPROCESS = {
    'simple_cnn': normalize_simple,
    'complex_cnn': normalize_complex,
    'resnet50': resnet50.preprocess_input,
    'mobilenetv2': mobilenet_v2.preprocess_input,
}


def main():
    class_names = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    if not class_names:
        raise ValueError('Không có thư mục lớp trong data/train')
    if not IMAGE_PATH:
        test_names = sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
        if class_names != test_names:
            raise ValueError('Tên lớp trong train và test không khớp')
        if TEST_STEPS is not None and TEST_STEPS < 1:
            raise ValueError('TEST_STEPS phải lớn hơn 0 hoặc bằng None')
    for filename in MODEL_FILES.values():
        if not filename.is_file():
            raise FileNotFoundError(f'Chưa có file model: {filename}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    scope = 'Toàn bộ tập test' if TEST_STEPS is None else f'Chạy thử {TEST_STEPS} batch đầu'

    for name, filename in MODEL_FILES.items():
        print('\nModel:', name, flush=True)

        # Load model từ file, không train lại
        model = load_saved_model(filename)
        if model.output_shape[-1] != len(class_names):
            raise ValueError(f'{name}: số nhãn của model không khớp dữ liệu')

        if IMAGE_PATH:
            image = load_img(IMAGE_PATH, target_size=(224, 224), interpolation='nearest')
            image = img_to_array(image)
            image = PREPROCESS[name](image)
            X = np.expand_dims(image, axis=0)
            y_prob = model.predict(X)
            y_pred = np.argmax(y_prob, axis=1)
            print('Nhãn dự đoán:', class_names[y_pred[0]])
            print('Confidence: %.2f%%' % (np.max(y_prob) * 100))
        else:
            # Đọc tập test, tắt shuffle để nhãn dự đoán khớp với nhãn thật
            test_datagen = ImageDataGenerator(preprocessing_function=PREPROCESS[name])
            test_data = test_datagen.flow_from_directory(
                TEST_DIR, target_size=(224, 224), batch_size=16,
                classes=class_names, class_mode='categorical', shuffle=False
            )
            if not test_data.samples:
                raise ValueError('Không có ảnh trong tập test')

            # Dự đoán nhãn của tập test
            steps = min(TEST_STEPS, len(test_data)) if TEST_STEPS is not None else None
            y_prob = model.predict(test_data, steps=steps)
            y_pred = np.argmax(y_prob, axis=1)
            y_test = test_data.classes[:len(y_pred)]

            # Đánh giá chất lượng dự đoán
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=range(len(class_names)), average='macro', zero_division=0
            )
            print(scope, '- số ảnh:', len(y_test))
            print('Accuracy: %.4f' % accuracy)
            print('Precision: %.4f' % precision)
            print('Recall: %.4f' % recall)
            print('F1-score: %.4f' % f1)
            results.append([name, len(y_test), accuracy, precision, recall, f1, scope])

            report = classification_report(
                y_test, y_pred, labels=range(len(class_names)),
                target_names=class_names, digits=4, zero_division=0
            )
            with open(OUTPUT_DIR / f'{name}_report.txt', 'w', encoding='utf-8') as f:
                f.write(scope + '\n\n' + report)
            with open(OUTPUT_DIR / f'{name}_predictions.csv', 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'true_label', 'predicted_label'])
                for path, actual, predicted in zip(test_data.filenames, y_test, y_pred):
                    writer.writerow([path, class_names[actual], class_names[predicted]])

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    if results:
        print('\nBẢNG SO SÁNH -', scope)
        print('%-20s %10s %10s %10s %10s' % ('Model', 'Accuracy', 'Precision', 'Recall', 'F1'))
        for name, count, accuracy, precision, recall, f1, scope in results:
            print('%-20s %10.4f %10.4f %10.4f %10.4f' % (name, accuracy, precision, recall, f1))
        with open(OUTPUT_DIR / 'model_comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'images', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'scope'])
            writer.writerows(results)
    print('\nĐã predict xong.', flush=True)


if __name__ == '__main__':
    main()
