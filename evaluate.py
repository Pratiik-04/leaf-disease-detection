import argparse
from pathlib import Path

import tensorflow as tf

from src.config import MODEL_PATH
from src.dataset import load_labeled_test_data


def evaluate(model_path: Path, data_dir: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    test_ds, class_names = load_labeled_test_data(data_dir)

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Classes: {', '.join(class_names)}")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained leaf disease model.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/validation"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(model_path=args.model, data_dir=args.data_dir)

