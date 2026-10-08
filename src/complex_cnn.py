"""Train Complex CNN on the prepared PlantVillage train/val splits."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "complex_cnn_best.keras"
OUTPUT_DIR = ROOT / "outputs" / "complex_cnn"
HISTORY_PATH = OUTPUT_DIR / "complex_cnn_history.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

TRAIN_MEAN = np.array([0.4666, 0.4893, 0.4104], dtype=np.float32)
TRAIN_STD = np.array([0.1992, 0.1750, 0.2174], dtype=np.float32)


def zscore_normalize(image):
    image = image / 255.0
    return (image - TRAIN_MEAN) / TRAIN_STD


def build_model():
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    for filters in [32, 64, 128, 256]:
        model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(38, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    for split in ["train", "val"]:
        if not (DATA_DIR / split).is_dir():
            raise FileNotFoundError(f"Missing data/{split} directory")

    # Protect the checkpoint saved from the previous training run.
    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Model already exists: {MODEL_PATH}. "
            "Back it up or rename it before starting a new run."
        )

    tf.keras.utils.set_random_seed(42)

    train_datagen = ImageDataGenerator(
        preprocessing_function=zscore_normalize,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=zscore_normalize)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_generator = val_datagen.flow_from_directory(
        DATA_DIR / "val",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    if train_generator.class_indices != val_generator.class_indices:
        raise ValueError("Train and validation class mappings do not match")
    if train_generator.num_classes != 38:
        raise ValueError("Expected 38 PlantVillage classes")

    model = build_model()
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False)
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
