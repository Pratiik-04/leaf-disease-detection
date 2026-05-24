import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import CLASS_NAMES_PATH, MODEL_PATH, OUTPUT_DIR
from src.preprocessing import list_images, prepare_image_for_model
from src.utils import ensure_dir, load_class_names


def predict_image(
    model: tf.keras.Model,
    image_path: Path,
    class_names: list[str],
) -> dict[str, object]:
    image_batch = prepare_image_for_model(image_path)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "image": str(image_path),
        "predicted_class": class_names[predicted_index],
        "confidence": round(float(probabilities[predicted_index]) * 100, 2),
    }


def predict_paths(
    model_path: Path,
    class_names_path: Path,
    image_paths: list[Path],
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_names_path)

    results = [
        predict_image(model=model, image_path=image_path, class_names=class_names)
        for image_path in image_paths
    ]
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict plant leaf disease from images.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Path to one image.")
    input_group.add_argument("--image-dir", type=Path, help="Path to a folder of images.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--classes", type=Path, default=CLASS_NAMES_PATH)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = list_images(args.image_dir)

    if not image_paths:
        raise ValueError("No supported images found for prediction.")

    predictions = predict_paths(
        model_path=args.model,
        class_names_path=args.classes,
        image_paths=image_paths,
    )

    print(predictions.to_string(index=False))

    if args.csv:
        ensure_dir(args.csv.parent)
        predictions.to_csv(args.csv, index=False)
        print(f"Predictions saved to: {args.csv}")
    elif args.image_dir:
        ensure_dir(OUTPUT_DIR)
        default_csv = OUTPUT_DIR / "predictions.csv"
        predictions.to_csv(default_csv, index=False)
        print(f"Predictions saved to: {default_csv}")


if __name__ == "__main__":
    main()

