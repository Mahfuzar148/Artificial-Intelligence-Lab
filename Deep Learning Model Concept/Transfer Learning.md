

* ✅ Pre-trained VGG16 (ImageNet)
* ✅ include_top=False
* ✅ Base model freeze
* ✅ Custom classifier add
* ✅ Compile
* ✅ Train
* ✅ Evaluate

---

# 📘 Transfer Learning Full Code Example (VGG16)

---

## 🟢 Step 1: Import Libraries

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
```

---

## 🟢 Step 2: Load Example Data (Dummy Example)

এখানে demonstration এর জন্য random data ব্যবহার করছি।

```python
# Dummy dataset (100 images)
x_train = np.random.rand(100, 224, 224, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, 5, 100), 5)

x_test = np.random.rand(20, 224, 224, 3)
y_test = tf.keras.utils.to_categorical(np.random.randint(0, 5, 20), 5)
```

---

## 🟢 Step 3: Load Pre-trained VGG16

```python
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

---

### 🔎 Explanation:

* weights='imagenet' → Pre-trained weights load
* include_top=False → Fully connected layer বাদ
* এখন শুধুমাত্র convolutional feature extractor থাকবে

---

## 🟢 Step 4: Freeze Base Model

```python
base_model.trainable = False
```

### কেন?

Pre-trained weights যেন পরিবর্তন না হয়।

---

## 🟢 Step 5: Add Custom Classifier

```python
inputs = base_model.input
x = base_model.output

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)

model = Model(inputs, outputs)
```

---

### 🔎 Structure এখন এমন:

```
Input → VGG16 Conv Blocks → Flatten → Dense → Output
```

---

## 🟢 Step 6: Compile Model

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🟢 Step 7: Train Model

```python
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2
)
```

---

## 🟢 Step 8: Evaluate Model

```python
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)
```

---

# 🔥 Complete Final Code (All Together)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dummy Data
x_train = np.random.rand(100, 224, 224, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, 5, 100), 5)

x_test = np.random.rand(20, 224, 224, 3)
y_test = tf.keras.utils.to_categorical(np.random.randint(0, 5, 20), 5)

# Load Pre-trained Model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze Base
base_model.trainable = False

# Add Custom Layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)

model = Model(base_model.input, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2
)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)
```

---

# 🧠 Transfer Learning Concept Summary

### 🔹 Phase 1 → Feature Extraction

* Base model freeze
* Only new Dense layers train

### 🔹 Phase 2 → Fine-tuning (Optional)

* কিছু convolution layer unfreeze করা
* Small learning rate দিয়ে train

---

# 🔵 Fine-Tuning Example (Advanced)

```python
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

# 🎯 When To Use Transfer Learning?

* Dataset ছোট হলে
* Faster training চাইলে
* High accuracy দরকার হলে

---

# 📝 Viva Ready Answer

> Transfer learning হলো এমন একটি technique যেখানে pre-trained model-এর convolutional অংশ ব্যবহার করে নতুন dataset-এর জন্য custom classifier যোগ করা হয়। এতে training time কম লাগে এবং accuracy ভালো হয়।

---

