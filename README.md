# Leaf Disease Detection Using Machine Learning

Python project for detecting plant leaf diseases from images using TensorFlow, OpenCV, PIL, NumPy, and Pandas.

This version is suitable for a college project report or resume demonstration. It supports:

- Training a CNN model from a folder-based image dataset
- Saving the trained model and class labels
- Predicting disease names for one image or a folder of preset test images
- Exporting prediction results to CSV

## Project Structure

```text
leaf-disease-detection/
├── dataset/
│   ├── train/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___healthy/
│   │   └── ...
│   └── test/
│       ├── image1.jpg
│       └── image2.jpg
├── models/
├── outputs/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   └── utils.py
├── train.py
├── predict.py
├── evaluate.py
└── requirements.txt
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Format

Place your training images inside `dataset/train`, grouped by disease class folder:

```text
dataset/train/Tomato___Early_blight/image_1.jpg
dataset/train/Tomato___Late_blight/image_2.jpg
dataset/train/Tomato___healthy/image_3.jpg
```

The folder name becomes the disease label.

## Train the Model

```bash
python train.py --data-dir dataset/train --epochs 20
```

The trained model will be saved to:

```text
models/leaf_disease_model.keras
models/class_names.json
```

## Predict a Single Image

```bash
python predict.py --image dataset/test/leaf_1.jpg
```

## Predict Multiple Preset Images

```bash
python predict.py --image-dir dataset/test --csv outputs/predictions.csv
```

## Evaluate on a Labeled Test Dataset

If your test dataset is also class-folder based:

```bash
python evaluate.py --data-dir dataset/validation
```

## Notes for Project Report

- Model type: Convolutional Neural Network using transfer learning with MobileNetV2
- Image size: 224 x 224
- Libraries: TensorFlow, OpenCV, PIL, NumPy, Pandas
- Output: Disease class and confidence score
- Target use case: Helping farmers identify plant disease early and reduce crop loss

