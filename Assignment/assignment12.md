## 📝 Question 12

**Write a code by discussing the effect of different data augmentation techniques on your CNN based classifiers.**

---

# 📘 Objective

We will:

* Build a CNN classifier
* Train it **without augmentation**
* Train it **with different augmentation techniques**
* Compare accuracy & loss curves
* Discuss effect of augmentation

Dataset used: **CIFAR-10**

---

# 🧠 Why Data Augmentation?

Data augmentation:

* Increases dataset diversity
* Reduces overfitting
* Improves generalization
* Makes model robust to rotation, shift, flip, zoom

---

# 🔵 Step 1: Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
```

---

# 🔵 Step 2: Load Dataset

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train / 255.0
x_test = x_test / 255.0
```

---

# 🔵 Step 3: Build CNN Model

```python
def build_cnn():
    inputs = Input((32,32,3))
    
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

---

# 🟢 PART 1: Train WITHOUT Augmentation

```python
model_no_aug = build_cnn()

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history_no_aug = model_no_aug.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

loss_no_aug, acc_no_aug = model_no_aug.evaluate(x_test, y_test)
print("Test Accuracy WITHOUT Augmentation:", acc_no_aug)
```

---

# 🟣 PART 2: Train WITH Data Augmentation

We will use:

* Random Flip
* Random Rotation
* Random Zoom
* Random Contrast

---

## 🔹 Define Augmentation Layer

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
```

---

## 🔹 Build CNN with Augmentation

```python
def build_cnn_with_aug():
    inputs = Input((32,32,3))
    
    x = data_augmentation(inputs)
    
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

---

## 🔹 Train Model WITH Augmentation

```python
model_aug = build_cnn_with_aug()

history_aug = model_aug.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

loss_aug, acc_aug = model_aug.evaluate(x_test, y_test)
print("Test Accuracy WITH Augmentation:", acc_aug)
```

---

# 📊 Compare Accuracy Curves

```python
plt.plot(history_no_aug.history['val_accuracy'], label='No Aug')
plt.plot(history_aug.history['val_accuracy'], label='With Aug')
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.show()
```

---

# 📉 Compare Loss Curves

```python
plt.plot(history_no_aug.history['val_loss'], label='No Aug')
plt.plot(history_aug.history['val_loss'], label='With Aug')
plt.title("Validation Loss Comparison")
plt.legend()
plt.show()
```

---

# 🧠 Effect of Data Augmentation

| Technique       | Effect                         |
| --------------- | ------------------------------ |
| Horizontal Flip | Handles mirror images          |
| Rotation        | Handles camera angle variation |
| Zoom            | Handles scale variation        |
| Contrast        | Handles lighting change        |

---

# 📊 Expected Observation

| Model       | Training Accuracy | Validation Accuracy | Overfitting |
| ----------- | ----------------- | ------------------- | ----------- |
| Without Aug | High              | Lower               | More        |
| With Aug    | Slightly Lower    | Higher              | Less        |

---

# 🎯 Conclusion

Data augmentation:

* Reduces overfitting
* Improves generalization
* Makes CNN robust
* Acts like increasing dataset size

---

# 📝 Viva Ready Explanation

> Data augmentation improves CNN performance by artificially increasing dataset diversity. Techniques such as flipping, rotation, zooming, and contrast adjustment reduce overfitting and improve generalization capability.

---

