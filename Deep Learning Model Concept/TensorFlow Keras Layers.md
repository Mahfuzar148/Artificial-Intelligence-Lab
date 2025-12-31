# 📘 TensorFlow Keras Layers – Full Documentation 

---

# ✅ `tensorflow.keras.layers` – FULL IMPORT LIST (আগে শুধু তালিকা)

```python
from tensorflow.keras.layers import (
    Input, Dense, Flatten,
    Conv1D, Conv2D, Conv3D,
    MaxPooling1D, MaxPooling2D, MaxPooling3D,
    AveragePooling1D, AveragePooling2D, AveragePooling3D,
    GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,
    GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D,
    SimpleRNN, LSTM, GRU,
    Dropout, BatchNormalization,
    Activation, ReLU, LeakyReLU, PReLU, Softmax,
    Embedding, Masking,
    Reshape, Permute, RepeatVector, Lambda,
    Add, Multiply, Concatenate, Subtract, Average, Maximum, Minimum,
    GaussianNoise, GaussianDropout,
    DepthwiseConv2D, SeparableConv2D,
    TimeDistributed, Bidirectional
)
```




---

```python
from tensorflow.keras import layers
```

অথবা

```python
from tensorflow.keras.layers import ...
```

---

## 🔹 1. Core / Basic Layers

👉 সব ধরনের Neural Network-এ লাগে

### 1️⃣ Input

```python
Input(shape=(...))
```

**কখন ব্যবহার করবেন**

* Functional API ব্যবহার করলে
* Model এর input shape define করতে

```python
Input(shape=(28,28,1))
```

---

### 2️⃣ Dense (Fully Connected Layer)

```python
Dense(units, activation=None)
```

**কখন ব্যবহার করবেন**

* ANN
* CNN/RNN এর শেষ অংশে
* Classification / Regression

```python
Dense(128, activation='relu')
Dense(10, activation='softmax')
```

---

### 3️⃣ Flatten

```python
Flatten()
```

**কখন ব্যবহার করবেন**

* CNN output কে Dense layer এ পাঠানোর আগে

```python
Flatten()
```

---

## 🔹 2. Convolutional Layers (CNN)

👉 Image, video, signal processing

### 4️⃣ Conv1D

```python
Conv1D(filters, kernel_size)
```

**ব্যবহার**

* Time-series
* Audio
* 1D signal

---

### 5️⃣ Conv2D

```python
Conv2D(filters, kernel_size)
```

**ব্যবহার**

* Image classification
* Object detection

```python
Conv2D(32, (3,3), activation='relu')
```

---

### 6️⃣ Conv3D

```python
Conv3D(filters, kernel_size)
```

**ব্যবহার**

* Video
* 3D medical images (MRI, CT)

---

## 🔹 3. Pooling Layers

👉 Feature map ছোট করতে

### 7️⃣ MaxPooling1D / 2D / 3D

```python
MaxPooling2D(pool_size=(2,2))
```

**ব্যবহার**

* Important feature retain করে
* CNN এ খুব common

---

### 8️⃣ AveragePooling

```python
AveragePooling2D()
```

**ব্যবহার**

* Smooth feature extraction

---

### 9️⃣ Global Pooling

```python
GlobalAveragePooling2D()
GlobalMaxPooling2D()
```

**ব্যবহার**

* Flatten ছাড়াই CNN শেষ করতে
* Parameter কমাতে

---

## 🔹 4. Recurrent Layers (RNN Family)

👉 Sequence / time dependent data

### 🔟 SimpleRNN

```python
SimpleRNN(units)
```

**ব্যবহার**

* Basic sequence
* Short memory task

❌ Long dependency ভালো handle করতে পারে না

---

### 1️⃣1️⃣ LSTM

```python
LSTM(units, return_sequences=False)
```

**ব্যবহার**

* NLP
* Time series forecasting
* Speech recognition

✔ Long-term dependency handle করে

---

### 1️⃣2️⃣ GRU

```python
GRU(units)
```

**ব্যবহার**

* LSTM এর lightweight version
* Faster training

---

## 🔹 5. Regularization Layers

👉 Overfitting কমাতে

### 1️⃣3️⃣ Dropout

```python
Dropout(rate)
```

**ব্যবহার**

* Training এর সময় neuron randomly বন্ধ করে

```python
Dropout(0.5)
```

---

### 1️⃣4️⃣ BatchNormalization

```python
BatchNormalization()
```

**ব্যবহার**

* Training speed বাড়ায়
* Gradient stable রাখে

---

## 🔹 6. Activation Layers

👉 Custom activation control

### 1️⃣5️⃣ Activation

```python
Activation('relu')
```

---

### 1️⃣6️⃣ ReLU / LeakyReLU / PReLU

```python
ReLU()
LeakyReLU(alpha=0.1)
PReLU()
```

**কখন কোনটা**

* **ReLU** → Default
* **LeakyReLU** → Dead neuron problem
* **PReLU** → Learnable slope

---

## 🔹 7. Embedding Layer (NLP)

### 1️⃣7️⃣ Embedding

```python
Embedding(input_dim, output_dim)
```

**ব্যবহার**

* Text classification
* Word representation

```python
Embedding(10000, 128)
```

---

## 🔹 8. Reshaping & Utility Layers

### 1️⃣8️⃣ Reshape

```python
Reshape(target_shape)
```

---

### 1️⃣9️⃣ Permute

```python
Permute((2,1))
```

Axis reorder করতে

---

### 2️⃣0️⃣ RepeatVector

```python
RepeatVector(n)
```

RNN encoder-decoder এ

---

## 🔹 9. Merge / Combine Layers

👉 Multiple input/output model

### 2️⃣1️⃣ Add

```python
Add()
```

---

### 2️⃣2️⃣ Multiply

```python
Multiply()
```

---

### 2️⃣3️⃣ Concatenate

```python
Concatenate(axis=-1)
```

---

## 🔹 10. Noise & Masking Layers

### 2️⃣4️⃣ GaussianNoise

```python
GaussianNoise(stddev)
```

---

### 2️⃣5️⃣ Masking

```python
Masking(mask_value=0.0)
```

Variable length sequence এ

---

## 🔹 11. Advanced CNN Layers

### 2️⃣6️⃣ DepthwiseConv2D

MobileNet type model

---

### 2️⃣7️⃣ SeparableConv2D

Lightweight CNN

---

## 🔹 12. Wrapper Layers

### 2️⃣8️⃣ TimeDistributed

```python
TimeDistributed(Dense(64))
```

Sequence এর প্রতিটা timestep এ same layer

---

### 2️⃣9️⃣ Bidirectional

```python
Bidirectional(LSTM(64))
```

Forward + backward context

---

## 🔹 13. সব একসাথে Import

```python
from tensorflow.keras import layers
```

---

## 🧠 কোন Problem → কোন Layer?

| Problem     | Layer              |
| ----------- | ------------------ |
| Image       | Conv2D, MaxPooling |
| NLP         | Embedding, LSTM    |
| Time series | LSTM, GRU          |
| Overfitting | Dropout            |
| Multi-input | Concatenate        |
| Fast CNN    | SeparableConv2D    |

---

## 🔥 Complete Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

