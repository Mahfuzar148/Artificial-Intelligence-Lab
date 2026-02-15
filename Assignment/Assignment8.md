## 📝 Question

**Build a Convolutional Neural Network (CNN) based classifier having architecture similar to the classical VGG16.**

---

# 📘 CNN Classifier Similar to Classical VGG16

### 🔹 What does “similar to VGG16” mean?

Classical **VGG16** architecture characteristics:

* Small filters (3×3)
* Multiple Conv layers stacked together
* After each block → MaxPooling
* Fully connected layers at the end
* Softmax output for classification

We will build a **VGG-like CNN (VGG-style architecture)** for 10-class classification.

---

# 🔵 Full Code (VGG-like CNN for CIFAR-10)

This example uses CIFAR-10 dataset.

---

```python
# ==========================================
# VGG-like CNN Classifier (10 Classes)
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Load CIFAR-10 Dataset
# ======================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# ======================
# Build VGG-like Model
# ======================

inputs = Input((32,32,3))

# -------- Block 1 --------
x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# -------- Block 2 --------
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# -------- Block 3 --------
x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# -------- Block 4 --------
x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# -------- Classification Head --------
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs, name="VGG_like_CNN")

# ======================
# Compile Model
# ======================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# EarlyStopping
# ======================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ======================
# Train
# ======================

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# ======================
# Evaluate
# ======================

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# ======================
# Plot Accuracy Curve
# ======================

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ======================
# Plot Loss Curve
# ======================

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

---

# 📘 Architecture Summary

VGG-style pattern:

```
[Conv → Conv] → MaxPool
[Conv → Conv] → MaxPool
[Conv → Conv → Conv] → MaxPool
[Conv → Conv → Conv] → MaxPool
Flatten
Dense
Dense
Output
```

---

# 🔬 Why This is VGG-like?

✔ Uses 3×3 filters
✔ Stacked convolution layers
✔ Block structure
✔ MaxPooling after each block
✔ Fully connected head

---

# 📊 Expected Performance

CIFAR-10 Accuracy ≈ 75–85% (depending on training time & hardware)

---

# 🧠 Viva Ready Explanation

> A VGG-like CNN classifier was implemented using stacked 3×3 convolutional layers grouped into blocks followed by max pooling layers. The architecture mimics classical VGG16 design principles and uses a fully connected classification head for 10-class prediction.

---


