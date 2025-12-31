
---

# 📘 `Sequential()` Model – Full Documentation (Keras / TensorFlow)

---

## 🔹 1. `Sequential()` কী?

`Sequential()` হলো Keras-এর **সবচেয়ে simple model API**, যেখানে:

* Layers একটার পর একটা **straight line** এ বসে
* Model-এর মধ্যে **কোন branching নেই**
* Single input → Single output

📌 নামই বলে দেয় → **Sequential = ধারাবাহিক (one-by-one)**

---

## 🔹 2. কখন `Sequential()` ব্যবহার করা যাবে?

### ✅ Use `Sequential()` যখন:

| Condition               | Allowed |
| ----------------------- | ------- |
| Single input            | ✅       |
| Single output           | ✅       |
| One layer after another | ✅       |
| No skip connection      | ✅       |
| No multi-branch         | ✅       |

📌 80% beginner + production model এখানেই হয়

---

## 🔹 3. কখন `Sequential()` ব্যবহার করা যাবে না?

### ❌ Do NOT use `Sequential()` যখন:

| Case                      | Reason              |
| ------------------------- | ------------------- |
| Multiple inputs           | Need Functional API |
| Multiple outputs          | Need Functional API |
| Skip connections (ResNet) | Graph structure     |
| Shared layers             | Not linear          |

---

## 🔹 4. Import Section

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
```

---

## 🔹 5. `Sequential()` ব্যবহার করার ২টা সঠিক উপায়

---

### ✅ Method-1: `.add()` দিয়ে (Beginner Friendly)

```python
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dense(1, activation='sigmoid'))
```

---

### ✅ Method-2: List দিয়ে (Clean & Professional)

```python
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])
```

---

## 🔹 6. Input Shape Rules ⭐ (Exam Important)

### 🔸 Dense layer হলে:

```python
input_shape = (features,)
```

### 🔸 Image হলে:

```python
input_shape = (height, width, channels)
```

📌 `input_shape` **শুধু প্রথম layer-এ** দিতে হয়

---

## 🔹 7. Common Use Cases with Examples

---

## 1️⃣ Regression Model

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dense(1)
])
```

📌 Output activation = linear (default)

---

## 2️⃣ Binary Classification

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(1, activation='sigmoid')
])
```

📌 Loss: `binary_crossentropy`

---

## 3️⃣ Multi-Class Classification

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(3, activation='softmax')
])
```

📌 Loss: `categorical_crossentropy`

---

## 4️⃣ Image Classification (FCNN)

```python
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## 5️⃣ CNN with Sequential

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    Flatten(),
    Dense(10, activation='softmax')
])
```

---

## 🔹 8. Compile + Train with Sequential

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=32)
```

---

## 🔹 9. `model.summary()` Output বুঝবো কীভাবে?

```python
model.summary()
```

Shows:

* Layer name
* Output shape
* Number of parameters
* Trainable / non-trainable

---

## 🔹 10. Common Mistakes ❌

### ❌ Wrong

```python
Sequential(Dense(), Dense())
```

### ✅ Correct

```python
Sequential([Dense(32), Dense(10)])
```

---

### ❌ Missing units

```python
Dense()  # WRONG
```

### ✅

```python
Dense(32)
```

---

### ❌ input_shape repeated

```python
Dense(32, input_shape=(10,))
Dense(16, input_shape=(10,))  # WRONG
```

---

## 🔹 11. Sequential vs Functional API (Quick Table)

| Feature         | Sequential | Functional |
| --------------- | ---------- | ---------- |
| Single input    | ✅          | ✅          |
| Multiple input  | ❌          | ✅          |
| Skip connection | ❌          | ✅          |
| Simplicity      | ⭐⭐⭐        | ⭐⭐         |

---

## 🔹 12. When to Switch from Sequential to Functional?

📌 Rule:

> If your model **cannot be drawn as a straight line**, don’t use Sequential.

---

## 🔹 13. Minimal Working Example

```python
model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')
])

model.summary()
```

---

## 🔹 14. Exam / Interview One-Liner ⭐

> **`Sequential()` API is used for linear stack models with single input and single output.**

---

## ✅ Final Summary

* `Sequential()` is the simplest Keras model API
* Best for beginners
* Fast to write & understand
* Limited for complex architectures

---

