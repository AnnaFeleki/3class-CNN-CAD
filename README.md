# 🧠 3D Convolutional Neural Network for Volumetric Data Classification

This project implements a 3D Convolutional Neural Network (3D CNN) using TensorFlow/Keras to classify volumetric (3D) medical imaging data such as CT or MRI scans. Ideal for exploring deep learning applications in healthcare, radiology, and computer vision.

---

## 🛠️ Requirements

Install the dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn

## 🧠 Model Architecture

```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    Conv3D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 64, 1)),
    MaxPooling3D(pool_size=2),
    BatchNormalization(),

    Conv3D(64, kernel_size=3, activation='relu'),
    MaxPooling3D(pool_size=2),
    BatchNormalization(),

    Conv3D(128, kernel_size=3, activation='relu'),
    MaxPooling3D(pool_size=2),
    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # For binary classification
])
```
