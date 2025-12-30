
---

# 🧾 `model.compile()` — Full Documentation (Keras)

## 🔹 `model.compile()` কী?

👉 `model.compile()` হলো সেই ধাপ যেখানে তুমি model-কে বলো:

> “আমি কীভাবে শিখবো,
> ভুল কীভাবে মাপবো,
> আর কী কী performance দেখাবো”

📌 **compile ছাড়া model train করা যায় না**।

---

## 🔹 Basic Syntax

```python
model.compile(
    optimizer,
    loss,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=None,
    jit_compile=False
)
```

---

# 🔴 Mandatory Parameters (অবশ্যই লাগবে)

## 1️⃣ `optimizer` ✅ (REQUIRED)

### 🔹 কাজ কী?

👉 Weight update করার নিয়ম ঠিক করে
(backpropagation + gradient descent)

---

### 🔹 কী কী value হতে পারে?

#### ▶️ String form (সবচেয়ে common)

```python
optimizer='adam'
optimizer='sgd'
optimizer='rmsprop'
optimizer='adagrad'
```

---

#### ▶️ Object form (Advanced / Recommended)

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
```

---

### 🔹 Common Optimizers Table

| Optimizer | Use-case                    |
| --------- | --------------------------- |
| `adam`    | Default, fast, most used    |
| `sgd`     | Simple, controlled learning |
| `rmsprop` | RNN / sequence data         |
| `adagrad` | Sparse features             |

---

### 🔹 না দিলে কী হবে?

❌ Error আসবে

```text
ValueError: optimizer must be specified
```

---

## 2️⃣ `loss` ✅ (REQUIRED)

### 🔹 কাজ কী?

👉 Model কতটা ভুল করছে সেটা মাপে

---

### 🔹 Loss Function কীভাবে choose করবে?

#### ▶️ Regression

```python
loss='mse'        # Mean Squared Error
loss='mae'        # Mean Absolute Error
loss='huber'
```

---

#### ▶️ Binary Classification

```python
loss='binary_crossentropy'
```

---

#### ▶️ Multi-class Classification

```python
loss='categorical_crossentropy'
loss='sparse_categorical_crossentropy'
```

---

### 🔹 Function form

```python
from tensorflow.keras.losses import MeanSquaredError
loss = MeanSquaredError()
```

---

### 🔹 না দিলে কী হবে?

❌ Error আসবে

```text
ValueError: loss must be specified
```

---

# 🟡 Optional Parameters (কিন্তু খুব গুরুত্বপূর্ণ)

## 3️⃣ `metrics` (OPTIONAL কিন্তু RECOMMENDED)

⚠️ তোমার কোডে এখানে **spelling mistake আছে**
❌ `metrices`
✅ `metrics`

---

### 🔹 কাজ কী?

👉 Training / validation চলাকালীন **performance দেখায়**
(loss ছাড়াও)

📌 Metrics দিয়ে weight update হয় না

---

### 🔹 Syntax

```python
metrics=['mae']
metrics=['accuracy']
metrics=['accuracy', 'precision', 'recall']
```

---

### 🔹 Common Metrics

#### ▶️ Regression

```python
metrics=['mae', 'mse']
```

#### ▶️ Classification

```python
metrics=['accuracy']
```

---

### 🔹 Function form

```python
from tensorflow.keras.metrics import MeanAbsoluteError

metrics=[MeanAbsoluteError()]
```

---

### 🔹 না দিলে কী হবে?

✔ Training হবে
❌ শুধু loss দেখাবে, extra info থাকবে না

---

## 4️⃣ `loss_weights` (Multi-output model)

### 🔹 কাজ কী?

👉 একাধিক output থাকলে
কোন loss কতটা গুরুত্বপূর্ণ সেটা বলে দেয়

```python
loss_weights=[0.7, 0.3]
```

📌 Single output model এ লাগে না

---

## 5️⃣ `weighted_metrics`

### 🔹 কাজ কী?

👉 Sample weight apply করার পর metric calculate করে

Rare use-case

---

## 6️⃣ `run_eagerly`

### 🔹 কাজ কী?

👉 Debugging mode

```python
run_eagerly=True
```

| Value | Meaning             |
| ----- | ------------------- |
| False | Fast (default)      |
| True  | Slow but debuggable |

---

## 7️⃣ `steps_per_execution`

### 🔹 কাজ কী?

👉 কত step একসাথে execute হবে (performance tuning)

```python
steps_per_execution=10
```

Advanced use-case

---

## 8️⃣ `jit_compile` (Advanced)

### 🔹 কাজ কী?

👉 XLA compilation enable করে (speed)

```python
jit_compile=True
```

GPU/TPU advanced optimization

---

# ✅ তোমার Corrected Compile Code

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

---

# 🔍 Regression vs Classification Examples

## 🔹 Regression Model

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

---

## 🔹 Binary Classification Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 🔹 Multi-class Classification Model

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

# 🧠 Compile Workflow (Big Picture)

```text
compile()
   ↓
fit()
   ↓
evaluate()
   ↓
predict()
```

📌 compile ছাড়া fit চলবে না

---

# ⚠️ Common Mistakes (VERY IMPORTANT)

❌ `metrics` এর spelling ভুল
❌ loss ভুল problem-type এর জন্য
❌ optimizer change করার পর recompile না করা
❌ trainable change করে compile না করা

---

# 🧠 Interview-ready One-liners

* `optimizer` controls how weights update
* `loss` measures error
* `metrics` monitor performance
* `compile()` prepares the model for training

---

# 📌 Summary Table

| Parameter    | Mandatory | কাজ                 |
| ------------ | --------- | ------------------- |
| optimizer    | ✅         | Weight update       |
| loss         | ✅         | Error calculation   |
| metrics      | ❌         | Performance report  |
| loss_weights | ❌         | Multi-output weight |
| run_eagerly  | ❌         | Debug               |
| jit_compile  | ❌         | Speed               |

---

## 🏁 Final Takeaway

> **`model.compile()` defines the learning strategy of a neural network.**



