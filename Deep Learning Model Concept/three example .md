
* ✅ FCNN for Binary Classification
* ✅ FCNN for Multi-class (MNIST)
* ✅ CNN for Image Classification (MNIST)

All codes include:

* Model creation
* compile()
* fit()
* evaluate()
* predict()

You can copy and run directly 🚀

---

# 🟢 1️⃣ Full FCNN Example (Binary Classification)

### 🎯 Problem: Classify random 2D points into 2 classes

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Generate dummy dataset
X = np.random.rand(1000, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build FCNN
inputs = Input((2,))
x = Dense(8, activation='relu')(inputs)
x = Dense(4, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Predict
predictions = model.predict(x_test)
predicted_classes = (predictions > 0.5).astype(int)
print("Predicted:", predicted_classes[:10].flatten())
```

---

# 🔵 2️⃣ Full FCNN Example (Multi-class MNIST)

### 🎯 Problem: Digit classification (0–9)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dims
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build FCNN
inputs = Input((28, 28, 1))
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Predict
pred = model.predict(x_test)
classes = pred.argmax(axis=1)
print("Predicted digits:", classes[:10])
```

---

# 🟣 3️⃣ Full CNN Example (MNIST)

### 🎯 Problem: Digit classification using CNN

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dims
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN
inputs = Input((28, 28, 1))

x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(64, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("CNN Test Accuracy:", acc)

# Predict
pred = model.predict(x_test)
classes = pred.argmax(axis=1)
print("Predicted digits:", classes[:10])
```


Digit classification 

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ===============================
# Load Data
# ===============================
(x_train, y_train), (x_test, y_test) = load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dims
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ===============================
# Build CNN Model
# ===============================
inputs = Input((28, 28, 1))

x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(64, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# ===============================
# Compile
# ===============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# Train
# ===============================
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# ===============================
# Evaluate
# ===============================
loss, acc = model.evaluate(x_test, y_test)
print("CNN Test Accuracy:", acc)

# ===============================
# Predict
# ===============================
pred = model.predict(x_test)
pred_classes = pred.argmax(axis=1)
true_classes = y_test.argmax(axis=1)

# ===============================
# Show 10 Test Images with Prediction
# ===============================
plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    
    predicted = pred_classes[i]
    actual = true_classes[i]
    
    if predicted == actual:
        color = 'green'
    else:
        color = 'red'
    
    plt.title(f"P:{predicted} / T:{actual}", color=color)
    plt.axis('off')

plt.suptitle(f"Test Accuracy: {acc:.4f}", fontsize=14)
plt.show()

# ===============================
# Plot Accuracy Curve
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ===============================
# Plot Loss Curve
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

```
---

Digit Classification Using Sparse Categorical Crossentropy
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model

# ===============================
# Load Data
# ===============================
(x_train, y_train), (x_test, y_test) = load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dims
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ⚠️ NO one-hot encoding here
# y_train and y_test remain integers (0-9)

# ===============================
# Build CNN Model
# ===============================
inputs = Input((28, 28, 1))

x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(64, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# ===============================
# Compile (Sparse Version)
# ===============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # 🔥 Changed here
    metrics=['accuracy']
)

# ===============================
# Train
# ===============================
history = model.fit(
    x_train, y_train,     # 🔥 integer labels
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# ===============================
# Evaluate
# ===============================
loss, acc = model.evaluate(x_test, y_test)
print("CNN Test Accuracy:", acc)

# ===============================
# Predict
# ===============================
pred = model.predict(x_test)
pred_classes = pred.argmax(axis=1)

# 🔥 true_classes directly y_test
true_classes = y_test

# ===============================
# Show 10 Test Images with Prediction
# ===============================
plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    
    predicted = pred_classes[i]
    actual = true_classes[i]
    
    if predicted == actual:
        color = 'green'
    else:
        color = 'red'
    
    plt.title(f"P:{predicted} / T:{actual}", color=color)
    plt.axis('off')

plt.suptitle(f"Test Accuracy: {acc:.4f}", fontsize=14)
plt.show()

# ===============================
# Plot Accuracy Curve
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ===============================
# Plot Loss Curve
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

```




# 🎯 Concept Summary

| Model Type   | Use Case             | Accuracy |
| ------------ | -------------------- | -------- |
| FCNN         | Tabular / Simple     | Medium   |
| FCNN (Image) | Basic image learning | Lower    |
| CNN          | Image classification | Higher   |

---

