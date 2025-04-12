# 🧠 3D Convolutional Neural Network for Volumetric Data Classification

This project implements a 3D Convolutional Neural Network (3D CNN) using TensorFlow/Keras to classify volumetric (3D) medical imaging data such as CT or MRI scans. Ideal for exploring deep learning applications in healthcare, radiology, and computer vision.

---

## 🛠️ Requirements

Install the dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```
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
## 🚀 Training the Model
You can load the data and train the model like this:
```bash
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load volumes and labels
X, y = [], []
for label, folder in enumerate(['dataset/class_0', 'dataset/class_1']):
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            volume = np.load(os.path.join(folder, file))
            X.append(volume)
            y.append(label)

X = np.array(X)[..., np.newaxis]  # Add channel dimension
y = to_categorical(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=8)
```
## 📊 Evaluation
```bash
from sklearn.metrics import classification_report

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

y_pred = model.predict(X_test)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
```
## 📈 Training Curves
```bash
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('3D CNN Training Accuracy')
plt.legend()
plt.show()
```


