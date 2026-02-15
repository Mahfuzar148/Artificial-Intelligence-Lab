## 📝 Question

**Write code by discussing the effect of the following issues on the classifier’s performance:**

* ● Different activation functions in hidden layers
* ● Different loss functions

---

# 📘 Objective

We will:

1️⃣ Train CNN using different **activation functions**

* ReLU
* Tanh
* Sigmoid

2️⃣ Train CNN using different **loss functions**

* Sparse Categorical Crossentropy
* Categorical Crossentropy

Dataset used: **MNIST**

We will compare:

* Validation accuracy
* Loss curves
* Final test accuracy

---

---

# 🔵 Step 1: Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
```

---

# 🔵 Step 2: Load Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# For categorical_crossentropy
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
```

---

# 🔵 Step 3: Function to Build CNN (Flexible Activation)

```python
def build_cnn(activation_function):
    
    inputs = Input((28,28,1))
    
    x = Conv2D(32, (3,3), activation=activation_function)(inputs)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation=activation_function)(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation=activation_function)(x)
    
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model
```

---

# 🟣 PART 1: Effect of Different Activation Functions

---

## 🔹 ReLU

```python
model_relu = build_cnn('relu')

model_relu.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_relu = model_relu.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3)]
)

loss_relu, acc_relu = model_relu.evaluate(x_test, y_test)
print("ReLU Accuracy:", acc_relu)
```

---

## 🔹 Tanh

```python
model_tanh = build_cnn('tanh')

model_tanh.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_tanh = model_tanh.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3)]
)

loss_tanh, acc_tanh = model_tanh.evaluate(x_test, y_test)
print("Tanh Accuracy:", acc_tanh)
```

---

## 🔹 Sigmoid

```python
model_sigmoid = build_cnn('sigmoid')

model_sigmoid.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_sigmoid = model_sigmoid.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3)]
)

loss_sigmoid, acc_sigmoid = model_sigmoid.evaluate(x_test, y_test)
print("Sigmoid Accuracy:", acc_sigmoid)
```

---

# 📊 Compare Activation Performance

```python
plt.plot(history_relu.history['val_accuracy'], label='ReLU')
plt.plot(history_tanh.history['val_accuracy'], label='Tanh')
plt.plot(history_sigmoid.history['val_accuracy'], label='Sigmoid')
plt.title("Activation Function Comparison")
plt.legend()
plt.show()
```

---

# 🧠 Expected Observation

| Activation | Performance | Reason                      |
| ---------- | ----------- | --------------------------- |
| ReLU       | Highest     | Avoids vanishing gradient   |
| Tanh       | Moderate    | Zero-centered but saturates |
| Sigmoid    | Lowest      | Strong vanishing gradient   |

---

---

# 🔴 PART 2: Effect of Different Loss Functions

---

## 🔹 Sparse Categorical Crossentropy

```python
model_sparse = build_cnn('relu')

model_sparse.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_sparse = model_sparse.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1
)

loss_sparse, acc_sparse = model_sparse.evaluate(x_test, y_test)
print("Sparse Loss Accuracy:", acc_sparse)
```

---

## 🔹 Categorical Crossentropy

```python
model_cat = build_cnn('relu')

model_cat.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_cat = model_cat.fit(
    x_train, y_train_cat,
    epochs=10,
    validation_split=0.1
)

loss_cat, acc_cat = model_cat.evaluate(x_test, y_test_cat)
print("Categorical Loss Accuracy:", acc_cat)
```

---

# 📊 Compare Loss Functions

```python
plt.plot(history_sparse.history['val_accuracy'], label='Sparse')
plt.plot(history_cat.history['val_accuracy'], label='Categorical')
plt.title("Loss Function Comparison")
plt.legend()
plt.show()
```

---

# 🧠 Expected Observation

| Loss Function | Label Type | Performance |
| ------------- | ---------- | ----------- |
| Sparse        | Integer    | Same        |
| Categorical   | One-hot    | Same        |

Difference mainly in **label encoding**, not performance.

---

# 🎯 Final Discussion

### 🔹 Activation Function Impact

* ReLU gives best performance
* Sigmoid suffers from vanishing gradient
* Tanh better than sigmoid but slower

### 🔹 Loss Function Impact

* Sparse & Categorical give similar accuracy
* Sparse simpler (no one-hot needed)

---

# 📝 Viva Ready Explanation

> Activation functions significantly impact classifier performance. ReLU generally performs best due to avoiding vanishing gradient problems. Different loss functions such as sparse categorical crossentropy and categorical crossentropy produce similar results, with differences mainly in label encoding format.

---
