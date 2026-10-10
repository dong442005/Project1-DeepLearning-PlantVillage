from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Cấu hình các tham số
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "mobilenetv2"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 10

# Data augmentation và preprocessing
# Dùng preprocess_input của MobileNetV2
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load data từ thư mục
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = test_val_datagen.flow_from_directory(
    'data/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# Tính Class Weights cho dữ liệu mất cân bằng
train_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights_array))

# Build model MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # đóng băng toàn bộ base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train Phase 1
print("\n--- Phase 1: Train classifier (base đóng băng) ---")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_1,
    validation_data=val_generator,
    class_weight=class_weight_dict
)

# Fine-tuning
print("\n--- Phase 2: Fine-tuning (mở băng 20 layer cuối) ---")
base_model.trainable = True

# MobileNetV2 nhẹ hơn ResNet50 nên mình mở ít layer hơn (20 so với 30 của anh A)
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # lr nhỏ xíu để fine-tune
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_2,
    validation_data=val_generator,
    class_weight=class_weight_dict
)

model.save('mobilenetv2_finetuned_model.h5')

# Evaluate trên tập Test
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# Vẽ đồ thị Accuracy
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

plt.figure(figsize=(8, 6))
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.axvline(x=EPOCHS_PHASE_1 - 1, color='r', linestyle='--', label='Bắt đầu Fine-tuning')
plt.title('Đồ thị Accuracy - MobileNetV2')
plt.legend()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / 'mobilenetv2_accuracy.png')
plt.show()

# Vẽ thêm đồ thị Loss
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.axvline(x=EPOCHS_PHASE_1 - 1, color='r', linestyle='--', label='Bắt đầu Fine-tuning')
plt.title('Đồ thị Loss - MobileNetV2')
plt.legend()
plt.savefig(OUTPUT_DIR / 'mobilenetv2_loss.png')
plt.show()

# 7️⃣ Classification Report & Confusion Matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix - MobileNetV2')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig(OUTPUT_DIR / 'mobilenetv2_confusion_matrix.png')
plt.show()

print("\nDONE!")
