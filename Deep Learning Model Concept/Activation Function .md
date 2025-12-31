
---

# 📘 Activation Function – Full Documentation (Deep Learning)

---

## 🔹 1. Activation Function কী?

**Activation function** হলো এমন একটি function যা:

* Neuron-এর output কে **non-linear** করে
* Neural Network-কে **complex pattern শেখার ক্ষমতা দেয়**

📌 Activation ছাড়া neural network শুধু **linear model** হয়।

---

## 🔹 2. Neuron Without vs With Activation

### ❌ Without Activation

[
y = Wx + b
]
→ Linear model
→ Deep হলেও shallow এর মতো behave করে

### ✅ With Activation

[
y = f(Wx + b)
]
→ Non-linear
→ Real-world problem solve করে

---

## 🔹 3. Activation Function কোথায় বসে?

```
Input → Dense → Activation → Output
```

বা

```python
Dense(64, activation='relu')
```

---

## 🔹 4. কেন Activation দরকার?

| Without Activation   | With Activation       |
| -------------------- | --------------------- |
| Only linear          | Non-linear            |
| XOR solve ❌          | XOR solve ✅           |
| Deep network useless | Deep network powerful |

---

# 🔥 5. Types of Activation Functions (সব গুরুত্বপূর্ণ)

---

## 1️⃣ Linear (No Activation)

### 📐 Formula

[
f(x) = x
]

### ✅ Keras

```python
Dense(1, activation='linear')
```

### 🔹 Use

* Regression
* Price / temperature prediction

### ⚠️ Note

* Hidden layer-এ ব্যবহার করা হয় না

---

## 2️⃣ Sigmoid

### 📐 Formula

[
f(x) = \frac{1}{1 + e^{-x}}
]

### 📊 Range

```
(0, 1)
```

### ✅ Keras

```python
Dense(1, activation='sigmoid')
```

### 🔹 Use

* Binary classification
* Probability output

### ❌ Problems

* Vanishing gradient
* Slow training

---

## 3️⃣ Tanh

### 📐 Formula

[
f(x) = \tanh(x)
]

### 📊 Range

```
(-1, 1)
```

### ✅ Keras

```python
Dense(64, activation='tanh')
```

### 🔹 Use

* Hidden layers (older models)
* RNN

### ❌ Problem

* Vanishing gradient

---

## 4️⃣ ReLU ⭐ (Most Popular)

### 📐 Formula

[
f(x) = \max(0, x)
]

### 📊 Range

```
[0, ∞)
```

### ✅ Keras

```python
Dense(64, activation='relu')
```

### 🔹 Use

* Hidden layers
* CNN / DNN

### ❌ Problem

* Dead neuron problem

---

## 5️⃣ Leaky ReLU

### 📐 Formula

[
f(x) =
\begin{cases}
x, & x>0 \
\alpha x, & x\le 0
\end{cases}
]

### ✅ Keras

```python
from tensorflow.keras.layers import LeakyReLU
LeakyReLU(alpha=0.1)
```

### 🔹 Use

* ReLU dead neuron problem solve করতে

---

## 6️⃣ PReLU

### 📐 Formula

* α **learnable**

### ✅ Keras

```python
from tensorflow.keras.layers import PReLU
PReLU()
```

### 🔹 Use

* When model needs adaptive slope

---

## 7️⃣ ELU

### 📐 Formula

* Smooth negative output

### ✅ Keras

```python
Dense(64, activation='elu')
```

### 🔹 Use

* Faster convergence than ReLU (some cases)

---

## 8️⃣ Softmax ⭐

### 📐 Formula

[
f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}
]

### 📊 Output

```
Sum of probabilities = 1
```

### ✅ Keras

```python
Dense(10, activation='softmax')
```

### 🔹 Use

* Multi-class classification (output layer only)

---

## 9️⃣ Swish

### 📐 Formula

[
f(x) = x \cdot \sigma(x)
]

### ✅ Keras

```python
Dense(64, activation='swish')
```

### 🔹 Use

* Modern deep networks
* EfficientNet

---

## 🔟 GELU

### 📐 Formula

* Gaussian based

### ✅ Keras

```python
Dense(64, activation='gelu')
```

### 🔹 Use

* Transformers
* NLP models (BERT)

---

# 🧠 6. Activation Selection Rule (Exam Important)

| Layer Type         | Best Activation |
| ------------------ | --------------- |
| Hidden layer       | ReLU            |
| Binary output      | Sigmoid         |
| Multi-class output | Softmax         |
| Regression output  | Linear          |
| RNN                | Tanh            |
| Transformer        | GELU            |

---

# ❌ 7. Common Mistakes

### ❌ Softmax in hidden layer

```python
Dense(64, activation='softmax')  # WRONG
```

---

### ❌ Sigmoid for multi-class

```python
Dense(3, activation='sigmoid')  # WRONG
```

---

### ❌ No activation at all

```python
Dense(64)  # Weak model
```

---

# 🧪 8. Activation Example Model

```python
inputs = Input((10,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)
```

---

# 📌 9. Activation Cheat Sheet

| Task        | Activation |
| ----------- | ---------- |
| Regression  | Linear     |
| Binary      | Sigmoid    |
| Multi-class | Softmax    |
| Hidden      | ReLU       |
| NLP         | GELU       |

---

## ✅ Final Summary

* Activation adds **non-linearity**
* Without activation → deep model useless
* ReLU is default for hidden layers
* Output activation depends on problem type

---

