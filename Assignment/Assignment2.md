

✅ 1) FCFNN Drawing (Conceptual Diagram)
✅ 2) Full TensorFlow.Keras Implementation
✅ 3) Training + Testing Example
✅ 4) Model Summary

আমি এখানে MNIST dataset ব্যবহার করছি।

---

# 📘 1️⃣ Drawing of Fully Connected Feed-Forward Neural Network (FCFNN)

### 🔹 My Preferred Architecture

* Input Layer → 784 neurons (28×28 image flattened)
* Hidden Layer 1 → 512 neurons
* Hidden Layer 2 → 256 neurons
* Hidden Layer 3 → 128 neurons
* Output Layer → 10 neurons (Softmax)

---

## 🧠 Network Diagram

```
                INPUT LAYER
        (784 neurons - flattened image)
                      │
                      ▼
            Hidden Layer 1 (512)
                 Activation: ReLU
                      │
                      ▼
            Hidden Layer 2 (256)
                 Activation: ReLU
                      │
                      ▼
            Hidden Layer 3 (128)
                 Activation: ReLU
                      │
                      ▼
             OUTPUT LAYER (10)
             Activation: Softmax
```

✔ Fully Connected
✔ Feed-Forward
✔ No Convolution

---

# 📘 2️⃣ Full Implementation Using TensorFlow.Keras

---

## 🔵 Complete Working Code

```python
# ==========================================
# 1️⃣ Import Libraries
# ==========================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist

# ==========================================
# 2️⃣ Load Dataset (MNIST)
# ==========================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ==========================================
# 3️⃣ Preprocessing
# ==========================================

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# ==========================================
# 4️⃣ Build FCFNN Model (Functional API)
# ==========================================

inputs = Input(shape=(28, 28))

# Flatten 28×28 → 784
x = Flatten()(inputs)

# Hidden Layers
x = Dense(512, activation='relu', name='hidden1')(x)
x = Dense(256, activation='relu', name='hidden2')(x)
x = Dense(128, activation='relu', name='hidden3')(x)

# Output Layer
outputs = Dense(10, activation='softmax', name='output')(x)

model = Model(inputs, outputs)

# ==========================================
# 5️⃣ Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 6️⃣ Show Model Summary
# ==========================================
model.summary()

# ==========================================
# 7️⃣ Train Model
# ==========================================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# ==========================================
# 8️⃣ Evaluate Model
# ==========================================

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# ==========================================
# 9️⃣ Plot Accuracy Curve
# ==========================================

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

---

# 📘 3️⃣ Model Architecture Explanation

```
Input (28×28 image)
↓
Flatten → 784 neurons
↓
Dense 512 (ReLU)
↓
Dense 256 (ReLU)
↓
Dense 128 (ReLU)
↓
Dense 10 (Softmax)
```

---

# 📊 Parameter Flow Example

First Hidden Layer:

[
(784 × 512) + 512 = 401,920
]

Total parameters ≈ 550K+

---

# 🧠 Why This is FCFNN?

✔ Every neuron connected to next layer
✔ Only forward propagation
✔ No convolution or pooling
✔ Used for structured/tabular/simple image classification

---

# 📝 Viva Ready Explanation

> A Fully Connected Feed-Forward Neural Network (FCFNN) was designed with three hidden layers (512, 256, and 128 neurons). The model was implemented using TensorFlow.Keras Functional API and trained on the MNIST dataset using sparse categorical crossentropy.

---

# 🎯 Expected Performance

MNIST Accuracy ≈ 97–98%

---


