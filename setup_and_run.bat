@echo off
setlocal

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete.
echo.
echo Next steps:
echo 1. Put your training images inside dataset\train\CLASS_NAME folders.
echo 2. Train the model:
echo    python train.py --data-dir dataset/train --epochs 20
echo 3. Put 10-12 test images inside dataset\test.
echo 4. Predict:
echo    python predict.py --image-dir dataset/test

