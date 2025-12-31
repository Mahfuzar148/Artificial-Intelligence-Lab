

---

# 📘 MNIST Digit Classification – Full Minimal Working Code (Keras)

---

## 🔹 Step 1: Import Required Libraries (Minimal)

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
```

---

## 🔹 Step 2: Load MNIST Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

📌 Dataset info:

* Image size: `28 × 28`
* Classes: `0–9` (10 classes)

---

## 🔹 Step 3: Preprocess Data (Mandatory)

### Normalize images

```python
x_train = x_train / 255.0
x_test  = x_test / 255.0
```

📌 Reason:

* Pixel range `0–255` → `0–1`
* Faster & stable training

---

## 🔹 Step 4: Build Model (Sequential, Minimal)

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```

📌 Explanation:

* `Flatten` → image → 1D vector (784)
* `Dense(32)` → hidden layer
* `Dense(10)` → 10 digit classes

---

## 🔹 Step 5: Compile Model (MANDATORY)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

📌 Minimum required:

* optimizer
* loss
* metrics (optional but practical)

---

## 🔹 Step 6: Train Model (fit)

```python
model.fit(
    x_train,
    y_train,
    epochs=5
)
```

📌 Minimum required:

* training data
* labels
* epochs

---

## 🔹 Step 7: Evaluate Model

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)
```

📌 Purpose:

* Model কতটা ভালো কাজ করছে তা দেখা

---

## 🔹 Step 8: Predict (Single Sample)

```python
predictions = model.predict(x_test)
```

### Predict one image

```python
import numpy as np

predicted_label = np.argmax(predictions[0])
true_label = y_test[0]

print("Predicted:", predicted_label)
print("Actual:", true_label)
```

📌 `argmax` → highest probability class

---

# 🧠 Minimal Parameters Summary (Exam Gold ⭐)

| Function       | Mandatory Parameters |
| -------------- | -------------------- |
| `Sequential()` | layers               |
| `Dense()`      | units                |
| `compile()`    | optimizer, loss      |
| `fit()`        | x, y, epochs         |
| `evaluate()`   | x, y                 |
| `predict()`    | x                    |

---

# 🧪 One-Block Complete Code (Copy–Paste Ready)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Predict
pred = model.predict(x_test)
print("Predicted label:", np.argmax(pred[0]))
print("Actual label:", y_test[0])
```

---

## ✅ Final Notes (Very Important)

* ✔ MNIST labels integer → `sparse_categorical_crossentropy`
* ✔ Softmax output units = number of classes
* ✔ Sequential best for this problem
* ✔ This is **minimum working deep learning pipeline**

---

