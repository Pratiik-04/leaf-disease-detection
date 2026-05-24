import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import CLASS_NAMES_PATH, MODEL_DIR, MODEL_PATH
from src.dataset import load_training_and_validation_data
from src.utils import ensure_dir, save_class_names


def build_model(num_classes: int, learning_rate: float) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="leaf_disease_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    data_dir: Path,
    epochs: int,
    validation_split: float,
    learning_rate: float,
) -> tf.keras.callbacks.History:
    train_ds, validation_ds, class_names = load_training_and_validation_data(
        data_dir=data_dir,
        validation_split=validation_split,
    )

    ensure_dir(MODEL_DIR)
    save_class_names(class_names, CLASS_NAMES_PATH)

    model = build_model(num_classes=len(class_names), learning_rate=learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leaf disease detection model.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
    )

