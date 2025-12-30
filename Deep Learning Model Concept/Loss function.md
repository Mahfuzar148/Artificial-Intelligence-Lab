

---

# 📘 LOSS FUNCTION — FULL DOCUMENTATION (Deep Learning)

---

## 1️⃣ Loss Function কী?

**Loss function** মাপে:

> Model যা predict করেছে (ŷ)
> আর আসল answer (y)
> এই দুইটার মধ্যে পার্থক্য কত

📌 Training এর লক্ষ্য:

```text
Loss minimize করা
```

Optimizer (SGD, Adam ইত্যাদি) এই loss কমানোর জন্য weight update করে।

---

## 2️⃣ Training Flow (Big Picture)

```
Input → Model → Prediction
              ↓
        Loss Function
              ↓
        Optimizer
              ↓
        Weight Update
```

---

## 3️⃣ Loss Function এর ধরন

### 🔹 A. Regression Loss

Continuous value predict করলে

| Loss  | কাজ                 |
| ----- | ------------------- |
| MSE   | Mean Squared Error  |
| MAE   | Mean Absolute Error |
| Huber | MSE + MAE mix       |

---

### 🔹 B. Classification Loss

Class predict করলে

| Problem                    | Output  | Loss                            |
| -------------------------- | ------- | ------------------------------- |
| Binary                     | sigmoid | binary_crossentropy             |
| Multiclass                 | softmax | categorical_crossentropy        |
| Multiclass (integer label) | softmax | sparse_categorical_crossentropy |

---

## 4️⃣ Categorical Crossentropy (DETAILS)

### 📌 ব্যবহার হবে যখন:

* Class > 2
* Output layer = `softmax`
* Label = **one-hot encoded**

---

### 🔹 Mathematical Formula

```text
L = − Σ yᵢ log(ŷᵢ)
```

* yᵢ = true label (0 or 1)
* ŷᵢ = predicted probability

👉 শুধু যেই class টা true (1), সেটার log probability নেওয়া হয়।

---

### 🔹 Example

True label:

```python
y_true = [0, 1, 0]
```

Prediction:

```python
y_pred = [0.1, 0.7, 0.2]
```

Loss:

```text
= −log(0.7)
= 0.357
```

✔ High probability → low loss
❌ Low probability → high loss

---

### 🔹 Keras ব্যবহার

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 5️⃣ Sparse Categorical Crossentropy

### 📌 পার্থক্য শুধু label format এ

| Type        | Example   |
| ----------- | --------- |
| categorical | [0, 1, 0] |
| sparse      | 1         |

### 🔹 ব্যবহার

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)
```

✔ এখানে `to_categorical` লাগবে না

---

## 6️⃣ Binary Crossentropy

### 📌 ব্যবহার হবে যখন:

* 2টা class
* Output = 1 neuron
* Activation = sigmoid

### 🔹 Formula

```text
L = −[ y log(p) + (1−y) log(1−p) ]
```

### 🔹 Keras

```python
Dense(1, activation='sigmoid')
loss='binary_crossentropy'
```

---

## 7️⃣ Regression Loss (সংক্ষেপে)

### 🔹 Mean Squared Error (MSE)

```text
L = (y − ŷ)²
```

✔ Large error কে বেশি শাস্তি দেয়
❌ Outlier sensitive

---

### 🔹 Mean Absolute Error (MAE)

```text
L = |y − ŷ|
```

✔ Robust to outliers
❌ Gradient constant → slow learning

---

### 🔹 Huber Loss

```text
Small error → MSE
Large error → MAE
```

✔ Best of both worlds

---

## 8️⃣ Loss vs Metric (Confusion দূর করো)

| Loss               | Metric                      |
| ------------------ | --------------------------- |
| Training এর জন্য   | Reporting এর জন্য           |
| Backpropagation হয় | Backprop হয় না              |
| Differentiable     | Non-differentiable হতে পারে |

Example:

```python
loss='categorical_crossentropy'
metrics=['accuracy']
```

---

## 9️⃣ Common Mistakes 🚨

### ❌ ভুল 1

```python
softmax + binary_crossentropy
```

### ❌ ভুল 2

```python
categorical_crossentropy + integer labels
```

### ❌ ভুল 3

```python
sigmoid + categorical_crossentropy
```

---

## 🔟 Correct Combination Cheat Sheet 🧠

| Output Layer      | Classes | Loss                            |
| ----------------- | ------- | ------------------------------- |
| Dense(1, sigmoid) | 2       | binary_crossentropy             |
| Dense(C, softmax) | C>2     | categorical_crossentropy        |
| Dense(C, softmax) | C>2     | sparse_categorical_crossentropy |

---

## 1️⃣1️⃣ Advanced Notes (Important)

### 🔹 Numerical Stability

Keras internally:

```text
softmax + crossentropy → fused implementation
```

👉 Overflow / underflow এড়ায়

---

### 🔹 Class Imbalance হলে

```python
class_weight = {0:1, 1:5}
model.fit(..., class_weight=class_weight)
```

---

## 🔑 One-line Summary

> **Loss function হলো model এর teacher — সে বলে দেয় “তুমি কতটা ভুল করছো”**

---

