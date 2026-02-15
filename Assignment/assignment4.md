

build an FCFNN based classifier according to your preferences about the 
number of hidden layers and neurons in the hidden layers. 

● training and testing your FCFNN based classifier using the: 
○ Fashion MNIST dataset. 

○ MNIST English dataset.

○ CIFAR-10 dataset.

---

# 🔵 1️⃣ MNIST – Deep FCFNN

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Load Data
# ======================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# ======================
# Build Deep FCFNN
# ======================
inputs = Input((784,))
x = Dense(1024, activation='relu')(inputs)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    x_train_flat, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# ======================
# Evaluate
# ======================
loss, acc = model.evaluate(x_test_flat, y_test)
print("MNIST Test Accuracy:", acc)

# ======================
# Predict + Show Images
# ======================
pred = model.predict(x_test_flat)
pred_classes = np.argmax(pred, axis=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"P:{pred_classes[i]} / T:{y_test[i]}")
    plt.axis('off')
plt.show()

# Accuracy Curve
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("MNIST Accuracy Curve")
plt.legend()
plt.show()

# Loss Curve
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("MNIST Loss Curve")
plt.legend()
plt.show()
```

---

# 🟣 2️⃣ Fashion MNIST – Deep FCFNN

```python
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

inputs = Input((784,))
x = Dense(1024, activation='relu')(inputs)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    x_train_flat, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(x_test_flat, y_test)
print("Fashion MNIST Test Accuracy:", acc)

pred = model.predict(x_test_flat)
pred_classes = np.argmax(pred, axis=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"P:{pred_classes[i]} / T:{y_test[i]}")
    plt.axis('off')
plt.show()

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Fashion Accuracy Curve")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Fashion Loss Curve")
plt.legend()
plt.show()
```

---

# 🔴 3️⃣ CIFAR-10 – Deep FCFNN

```python
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_flat = x_train.reshape(-1, 3072)
x_test_flat = x_test.reshape(-1, 3072)

inputs = Input((3072,))
x = Dense(2048, activation='relu')(inputs)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    x_train_flat, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(x_test_flat, y_test)
print("CIFAR-10 Test Accuracy:", acc)

pred = model.predict(x_test_flat)
pred_classes = np.argmax(pred, axis=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"P:{pred_classes[i]} / T:{y_test[i]}")
    plt.axis('off')
plt.show()

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("CIFAR Accuracy Curve")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("CIFAR Loss Curve")
plt.legend()
plt.show()
```

---

# 📊 Expected Accuracy

| Dataset       | Accuracy |
| ------------- | -------- |
| MNIST         | ~98%     |
| Fashion MNIST | ~90%     |
| CIFAR-10      | ~50–55%  |

---

# 🧠 Why More Dense Layers?

* More representation power
* Better feature abstraction
* But risk of overfitting

---

# 📝 Viva Ready Explanation

> A deep Fully Connected Feed-Forward Neural Network was implemented using multiple dense layers to increase representational capacity. EarlyStopping was applied to prevent overfitting, and prediction visualization was used to compare actual and predicted labels.

---

