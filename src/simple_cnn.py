"""
Thành viên B: Simple CNN (Baseline lấy cảm hứng từ LeNet-5) cho PlantVillage (38 lớp).

Chạy từ thư mục gốc của repo:
    python src/simple_cnn.py                 # train đầy đủ (mặc định 20 epochs)
    python src/simple_cnn.py --epochs 30
    python src/simple_cnn.py --smoke-test    # chạy thử nhanh toàn bộ pipeline, file kết quả có hậu tố _smoke
"""
import argparse
import os
import random
import sys
import time

import matplotlib
matplotlib.use('Agg')  # Chạy script không cần màn hình (Colab / terminal)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

# 1️⃣ Cấu hình chung (giống hệt data_prep.py của Thành viên A)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'
RESULTS_DIR = 'results'


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# 2️⃣ Chuẩn hóa Z-Score (copy từ docstring cuối file src/data_prep.py, Mean/Std tính trên tập Train)
def z_score_norm(img):
    img = img / 255.0
    mean = np.array([0.46658055, 0.48930454, 0.41043331])
    std = np.array([0.1992171,  0.17497431, 0.21742669])
    return (img - mean) / std


def check_data_dirs():
    missing = [d for d in (TRAIN_DIR, VAL_DIR, TEST_DIR) if not os.path.isdir(d)]
    if missing:
        sys.exit(
            f"[LỖI] Không tìm thấy thư mục: {', '.join(missing)}\n"
            f"Thư mục hiện tại: {os.getcwd()}\n"
            "Hãy chạy script từ thư mục gốc của repo (python src/simple_cnn.py) "
            "và đảm bảo đã có data/train, data/val, data/test."
        )


def build_generators():
    # Data Augmentation CHỈ cho tập Train
    train_datagen = ImageDataGenerator(
        preprocessing_function=z_score_norm,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    # Val/Test chỉ chuẩn hóa Z-score, không augmentation
    test_val_datagen = ImageDataGenerator(preprocessing_function=z_score_norm)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_generator = test_val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_generator = test_val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # [QUAN TRỌNG] Tắt shuffle để y_pred khớp thứ tự test_generator.classes
    )
    return train_generator, val_generator, test_generator


def compute_class_weight_dict(train_generator):
    # Class Weights 'balanced' vì dữ liệu mất cân bằng (~36x giữa lớp nhiều nhất và ít nhất)
    train_labels = train_generator.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    return dict(enumerate(class_weights_array))


# 3️⃣ Mô hình Simple CNN lấy cảm hứng từ LeNet-5 (KHÔNG BatchNorm, KHÔNG Dropout)
def build_simple_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        # Block 1: 224x224x3 -> Conv 5x5 -> 220x220x6 -> MaxPool -> 110x110x6
        layers.Conv2D(6, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Block 2: 110x110x6 -> Conv 5x5 -> 106x106x16 -> MaxPool -> 53x53x16
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Phân loại: Flatten (53*53*16 = 44,944) -> Dense 120 -> Dense 84 -> Softmax
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='simple_cnn')

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 4️⃣ Vẽ đồ thị Loss / Accuracy
def plot_training_curves(history, path):
    epochs_range = range(1, len(history['loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs_range, history['loss'], 'o-', label='Train Loss')
    ax_loss.plot(epochs_range, history['val_loss'], 'o-', label='Val Loss')
    ax_loss.set_title('Đồ thị Loss (Simple CNN)')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True)
    ax_loss.legend()

    ax_acc.plot(epochs_range, history['accuracy'], 'o-', label='Train Accuracy')
    ax_acc.plot(epochs_range, history['val_accuracy'], 'o-', label='Val Accuracy')
    ax_acc.set_title('Đồ thị Accuracy (Simple CNN)')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.grid(True)
    ax_acc.legend()

    fig.suptitle('Đồ thị Loss / Accuracy (Simple CNN)')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# 5️⃣ Vẽ Ma trận nhầm lẫn (Confusion Matrix)
def plot_confusion_matrix(cm, class_names, path):
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 7},
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Ma trận nhầm lẫn (Simple CNN)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Simple CNN (LeNet-5-inspired) trên PlantVillage')
    parser.add_argument('--epochs', type=int, default=20, help='Số epoch tối đa (mặc định 20)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Chạy thử nhanh: 5 bước train, 2 bước val, 1 epoch, 2 batch test; file có hậu tố _smoke')
    args = parser.parse_args()

    # In tiếng Việt không lỗi trên console Windows (cp1252)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')

    set_seeds()
    check_data_dirs()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix = '_smoke' if args.smoke_test else ''
    model_path = f'simple_cnn_model{suffix}.h5'
    curves_path = os.path.join(RESULTS_DIR, f'simple_cnn_training_curves{suffix}.png')
    cm_path = os.path.join(RESULTS_DIR, f'simple_cnn_confusion_matrix{suffix}.png')
    report_path = os.path.join(RESULTS_DIR, f'simple_cnn_classification_report{suffix}.txt')

    if args.smoke_test:
        epochs, steps_per_epoch, validation_steps, test_steps = 1, 5, 2, 2
    else:
        epochs, steps_per_epoch, validation_steps, test_steps = args.epochs, None, None, None

    train_generator, val_generator, test_generator = build_generators()
    num_classes = len(train_generator.class_indices)
    class_names = list(test_generator.class_indices.keys())
    class_weight_dict = compute_class_weight_dict(train_generator)

    model = build_simple_cnn(num_classes)
    model.summary()
    total_params = model.count_params()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    ]

    # 6️⃣ Huấn luyện
    start = time.time()
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    training_time = time.time() - start
    epochs_trained = len(history.history['loss'])
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1

    # EarlyStopping chỉ khôi phục best weights khi dừng sớm; nạp lại file checkpoint
    # để mô hình được đánh giá trùng với file .h5 nộp lên Drive (epoch có val_loss thấp nhất)
    model = load_model(model_path)

    plot_training_curves(history.history, curves_path)

    # 7️⃣ Đánh giá trên tập Test
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    Y_pred = model.predict(test_generator, steps=test_steps)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes[:len(y_pred)]

    labels = list(range(num_classes))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   digits=4, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, class_names, cm_path)

    summary = (
        f"===== SIMPLE CNN - KẾT QUẢ TRÊN TẬP TEST{' (SMOKE TEST)' if args.smoke_test else ''} =====\n"
        f"Số ảnh test được đánh giá : {len(y_true)}\n"
        f"Test loss                 : {test_loss:.4f}\n"
        f"Test accuracy (Keras)     : {test_acc:.4f}\n"
        f"Test accuracy (sklearn)   : {accuracy_score(y_true, y_pred):.4f}\n"
        f"Macro    Precision / Recall / F1 : {macro[0]:.4f} / {macro[1]:.4f} / {macro[2]:.4f}\n"
        f"Weighted Precision / Recall / F1 : {weighted[0]:.4f} / {weighted[1]:.4f} / {weighted[2]:.4f}\n"
        f"Tổng số tham số (params)  : {total_params:,}\n"
        f"Số epoch đã train         : {epochs_trained} (best epoch theo val_loss: {best_epoch})\n"
        f"Thời gian train           : {training_time:.1f} giây ({training_time / 60:.1f} phút)\n"
        f"File mô hình              : {model_path}\n"
    )
    full_report = summary + "\n===== CLASSIFICATION REPORT =====\n" + report

    print(full_report)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_report)

    print(f"Đã lưu: {curves_path}, {cm_path}, {report_path}, {model_path}")


if __name__ == '__main__':
    main()
