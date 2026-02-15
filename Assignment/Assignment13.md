## 📝 Question

**Show the effect of Dropout layer and Data Augmentation techniques on overfitting issues of your CNN based classifier.**

---

# 📘 Objective

We will:

1️⃣ Train a CNN **without Dropout & Augmentation**
2️⃣ Train a CNN **with Dropout only**
3️⃣ Train a CNN **with Dropout + Data Augmentation**

Then compare:

* Training Accuracy
* Validation Accuracy
* Loss Curves
* Overfitting behavior

Dataset: **CIFAR-10**

---

# 🔵 Step 1: Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
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

# 🔴 PART 1: CNN WITHOUT Dropout & Augmentation

```python
def build_basic_cnn():
    inputs = Input((32,32,3))
    
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
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

model_basic = build_basic_cnn()

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history_basic = model_basic.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)
```

---

# 🟣 PART 2: CNN WITH Dropout

```python
def build_dropout_cnn():
    inputs = Input((32,32,3))
    
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)   # 🔥 Dropout Layer
    
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model_dropout = build_dropout_cnn()

history_dropout = model_dropout.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)
```

---

# 🟢 PART 3: CNN WITH Dropout + Data Augmentation

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def build_aug_dropout_cnn():
    inputs = Input((32,32,3))
    
    x = data_augmentation(inputs)   # 🔥 Augmentation
    
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model_aug = build_aug_dropout_cnn()

history_aug = model_aug.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)
```

---

# 📊 Compare Accuracy Curves

```python
plt.plot(history_basic.history['val_accuracy'], label='Basic CNN')
plt.plot(history_dropout.history['val_accuracy'], label='With Dropout')
plt.plot(history_aug.history['val_accuracy'], label='Dropout + Aug')
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.show()
```

---

# 📉 Compare Loss Curves

```python
plt.plot(history_basic.history['val_loss'], label='Basic CNN')
plt.plot(history_dropout.history['val_loss'], label='With Dropout')
plt.plot(history_aug.history['val_loss'], label='Dropout + Aug')
plt.title("Validation Loss Comparison")
plt.legend()
plt.show()
```

---

# 🧠 Expected Observation

| Model         | Overfitting | Validation Accuracy |
| ------------- | ----------- | ------------------- |
| Basic CNN     | High        | Lower               |
| With Dropout  | Reduced     | Improved            |
| Dropout + Aug | Lowest      | Highest             |

---

# 🔬 Why Dropout Works?

Dropout:

* Randomly disables neurons during training
* Prevents co-adaptation
* Acts like ensemble learning

---

# 🔬 Why Augmentation Works?

Augmentation:

* Increases data diversity
* Makes model robust
* Reduces memorization

---

# 🎯 Final Conclusion

✔ Basic CNN → Overfits
✔ Dropout → Reduces overfitting
✔ Dropout + Augmentation → Best generalization

---

# 📝 Viva Ready Explanation

> Dropout reduces overfitting by randomly deactivating neurons during training, preventing co-adaptation. Data augmentation increases dataset diversity, improving generalization. Combining both techniques significantly reduces overfitting in CNN classifiers.

---

