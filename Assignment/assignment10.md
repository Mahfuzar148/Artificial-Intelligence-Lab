## 📝 Question

**Training a binary classifier based on the pre-trained VGG16 using transfer learning and fine-tuning.**

Also, showing the effect of fine-tuning:

i. Fine-tuning the whole pre-trained VGG16
ii. Fine-tuning partial layers of the pre-trained VGG16

Dataset: **Cats vs Dogs (Binary Classification)**

---

---

# 📘 Solution: Transfer Learning + Fine-Tuning using VGG16 (Cats vs Dogs)

We will:

* Use `tf.keras.applications.VGG16`
* Use Cats vs Dogs dataset from TensorFlow
* Build binary classifier
* Compare:

  * Partial fine-tuning
  * Full fine-tuning

---

---

# 🔵 Step 1: Import Libraries

```python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
```

---

# 🔵 Step 2: Load Cats vs Dogs Dataset

```python
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

path = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)

base_dir = path.replace('.zip', '')

train_dir = base_dir + '/train'
val_dir = base_dir + '/validation'

IMG_SIZE = (224,224)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
```

---

# 🔵 Step 3: Load Pre-trained VGG16

```python
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
```

---

# 🔵 Step 4: Add Custom Classification Head

```python
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, outputs)
```

---

# 🟣 CASE 1️⃣: Partial Fine-Tuning

Freeze most layers, unfreeze last few.

---

```python
# Freeze entire base model first
base_model.trainable = False

# Unfreeze last 4 layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history_partial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

loss_partial, acc_partial = model.evaluate(val_ds)
print("Partial Fine-Tuning Accuracy:", acc_partial)
```

---

# 🔴 CASE 2️⃣: Whole Fine-Tuning

Unfreeze entire VGG16.

---

```python
# Make all layers trainable
base_model.trainable = True

model.compile(
    optimizer=Adam(1e-6),  # Very small LR
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_full = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

loss_full, acc_full = model.evaluate(val_ds)
print("Full Fine-Tuning Accuracy:", acc_full)
```

---

# 🔵 Plot Accuracy Comparison

```python
plt.plot(history_partial.history['accuracy'], label='Partial FT')
plt.plot(history_full.history['accuracy'], label='Full FT')
plt.title("Fine-Tuning Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

---

# 📊 Expected Observation

| Fine-Tuning Type | Accuracy                         | Risk of Overfitting |
| ---------------- | -------------------------------- | ------------------- |
| Partial          | Stable                           | Low                 |
| Full             | Slightly Higher (if enough data) | High                |

---

# 🧠 Explanation of Effect

### 🔹 Partial Fine-Tuning

* Only last few layers updated
* Faster training
* Better for small dataset
* Preserves general ImageNet features

### 🔹 Full Fine-Tuning

* All layers updated
* More adaptable
* Requires small learning rate
* Needs larger dataset to avoid overfitting

---

# 📝 Viva Ready Explanation

> In partial fine-tuning, only the last few convolutional layers of VGG16 are unfrozen and trained while earlier layers remain frozen. In full fine-tuning, all convolutional layers are made trainable and updated with a very small learning rate. Full fine-tuning allows deeper adaptation but increases overfitting risk.

---

---
