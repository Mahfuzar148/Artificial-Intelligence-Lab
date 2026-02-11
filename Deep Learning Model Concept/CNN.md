

1. Basic CNN
2. LeNet-5
3. AlexNet
4. VGG16-style
5. ResNet (Residual Block সহ)
6. MobileNet (Depthwise Separable Conv)
7. U-Net (Segmentation)
8. 1D CNN
9. 3D CNN


* 🔹 Concept
* 🔹 Architecture explanation
* 🔹 When to use
* 🔹 Full model code

---

# 🟢 1️⃣ Basic CNN

## 🔹 Use Case

* MNIST
* Small dataset

## 🔹 Architecture

Conv → Pool → Conv → Pool → Flatten → Dense → Output

## 🔹 Full Code

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

inputs = Input((28,28,1))

x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🔵 2️⃣ LeNet-5

## 🔹 Developed by: Yann LeCun

## 🔹 Used for: Digit recognition

## 🔹 Architecture

Conv(6) → Pool → Conv(16) → Pool → FC → FC → Output

## 🔹 Full Code

```python
inputs = Input((32,32,1))

x = Conv2D(6, (5,5), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(16, (5,5), activation='relu')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(84, activation='relu')(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🔴 3️⃣ AlexNet

## 🔹 ImageNet 2012 Winner

## 🔹 Key Features

* Large filters
* ReLU
* Dropout

## 🔹 Full Code (Simplified)

```python
from tensorflow.keras.layers import Dropout

inputs = Input((227,227,3))

x = Conv2D(96, (11,11), strides=4, activation='relu')(inputs)
x = MaxPooling2D((3,3), strides=2)(x)

x = Conv2D(256, (5,5), activation='relu')(x)
x = MaxPooling2D((3,3), strides=2)(x)

x = Conv2D(384, (3,3), activation='relu')(x)
x = Conv2D(384, (3,3), activation='relu')(x)
x = Conv2D(256, (3,3), activation='relu')(x)
x = MaxPooling2D((3,3), strides=2)(x)

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(1000, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🟣 4️⃣ VGG16

## 🔹 Key Idea

* Multiple 3×3 Conv
* Very deep

## 🔹 Full Code (VGG Block Style)

```python
inputs = Input((224,224,3))

x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)

outputs = Dense(1000, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🟤 5️⃣ ResNet (With Residual Block)

## 🔹 Key Innovation

Skip connection: F(x) + x

## 🔹 Residual Block Code

```python
from tensorflow.keras.layers import Add

def residual_block(x, filters):
    shortcut = x
    
    x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    
    x = Add()([x, shortcut])
    x = tf.keras.activations.relu(x)
    
    return x

inputs = Input((64,64,3))
x = Conv2D(64, (3,3), padding='same')(inputs)

x = residual_block(x, 64)
x = residual_block(x, 64)

x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# ⚫ 6️⃣ MobileNet (Depthwise Separable Conv)

```python
from tensorflow.keras.layers import DepthwiseConv2D

inputs = Input((128,128,3))

x = DepthwiseConv2D((3,3), padding='same', activation='relu')(inputs)
x = Conv2D(64, (1,1), activation='relu')(x)

x = DepthwiseConv2D((3,3), padding='same', activation='relu')(x)
x = Conv2D(128, (1,1), activation='relu')(x)

x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🟢 7️⃣ U-Net (Segmentation)

```python
from tensorflow.keras.layers import UpSampling2D, Concatenate

inputs = Input((128,128,1))

# Encoder
c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2,2))(c1)

# Bottleneck
b = Conv2D(128, (3,3), activation='relu', padding='same')(p1)

# Decoder
u1 = UpSampling2D((2,2))(b)
concat = Concatenate()([u1, c1])
c2 = Conv2D(64, (3,3), activation='relu', padding='same')(concat)

outputs = Conv2D(1, (1,1), activation='sigmoid')(c2)

model = Model(inputs, outputs)
model.summary()
```

---

# 🔵 8️⃣ 1D CNN (Time Series)

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D

inputs = Input((100,1))

x = Conv1D(32, 3, activation='relu')(inputs)
x = MaxPooling1D(2)(x)

x = Flatten()(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🟡 9️⃣ 3D CNN (Video)

```python
from tensorflow.keras.layers import Conv3D, MaxPooling3D

inputs = Input((16,64,64,3))

x = Conv3D(32, (3,3,3), activation='relu')(inputs)
x = MaxPooling3D((2,2,2))(x)

x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

# 🎯 Final Understanding

| Model     | Main Idea              | Use Case             |
| --------- | ---------------------- | -------------------- |
| Basic CNN | Simple conv            | Small dataset        |
| LeNet     | Early digit model      | MNIST                |
| AlexNet   | Deep + dropout         | ImageNet             |
| VGG       | Small filters deep net | Large classification |
| ResNet    | Skip connection        | Very deep network    |
| MobileNet | Lightweight            | Mobile               |
| U-Net     | Encoder-decoder        | Segmentation         |
| 1D CNN    | Sequence data          | Time series          |
| 3D CNN    | Video                  | Action recognition   |

---

