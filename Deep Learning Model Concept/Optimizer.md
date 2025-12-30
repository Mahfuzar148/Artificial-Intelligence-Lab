
---

# 🧾 Optimizer — Full Documentation (TensorFlow / Keras)

---

## 🔹 Optimizer কী?

👉 **Optimizer** হলো সেই algorithm যা বলে দেয়:

> **loss কমানোর জন্য model-এর weight কীভাবে update হবে**

সহজভাবে:

```
Loss → Gradient → Optimizer → Weight update
```

📌 Optimizer ছাড়া neural network **শিখতেই পারবে না**।

---

## 🔹 Optimizer কী কাজ করে?

প্রতিটা training step-এ optimizer:

1️⃣ Loss calculate করে
2️⃣ Gradient বের করে (`∂loss/∂weight`)
3️⃣ Weight update করে

Formula (basic idea):

```
new_weight = old_weight − learning_rate × gradient
```

---

## 🔹 `model.compile()`-এ optimizer কীভাবে দেওয়া হয়?

### ▶️ String form (simple)

```python
model.compile(
    optimizer='adam',
    loss='mse'
)
```

### ▶️ Object form (recommended)

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='mse'
)
```

---

# 🔴 Core Parameter (সব optimizer-এ common)

## 🔑 `learning_rate` (সবচেয়ে গুরুত্বপূর্ণ)

### 🔹 learning rate কী?

👉 weight কত **দূরে যাবে** সেটা ঠিক করে

* ছোট → slow learning
* বড় → unstable / diverge

```python
Adam(learning_rate=0.001)
```

### 🔹 Wrong learning rate হলে কী হয়?

| LR      | Result             |
| ------- | ------------------ |
| খুব বড়  | Loss explode       |
| খুব ছোট | Training খুব slow  |
| ঠিক     | Smooth convergence |

---

# 🔹 Common Optimizers (সবচেয়ে বেশি ব্যবহৃত)

---

## 1️⃣ SGD (Stochastic Gradient Descent)

### 🔹 Concept

সবচেয়ে basic optimizer

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=False
)
```

### 🔹 Parameters

| Parameter       | কাজ                  |
| --------------- | -------------------- |
| `learning_rate` | Step size            |
| `momentum`      | Past gradient memory |
| `nesterov`      | Advanced momentum    |

### 🔹 Use-case

* Simple problems
* When you want full control

---

## 2️⃣ Adam (MOST POPULAR)

### 🔹 Concept

SGD + Momentum + RMSProp

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
```

### 🔹 Parameters

| Parameter       | কাজ                 |
| --------------- | ------------------- |
| `learning_rate` | Main step size      |
| `beta_1`        | 1st moment decay    |
| `beta_2`        | 2nd moment decay    |
| `epsilon`       | Numerical stability |

### 🔹 Why Adam is popular?

✔ Fast convergence
✔ Less tuning
✔ Default choice

---

## 3️⃣ RMSprop

### 🔹 Concept

Adaptive learning rate per parameter

```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9,
    epsilon=1e-7
)
```

### 🔹 Use-case

* RNN
* Sequence data

---

## 4️⃣ Adagrad

### 🔹 Concept

Learning rate decreases over time

```python
from tensorflow.keras.optimizers import Adagrad

optimizer = Adagrad(
    learning_rate=0.01,
    initial_accumulator_value=0.1
)
```

### 🔹 Problem

* LR খুব দ্রুত ছোট হয়ে যায়

---

## 5️⃣ Adamax

### 🔹 Concept

Adam-এর infinity-norm version

```python
from tensorflow.keras.optimizers import Adamax(
    learning_rate=0.002
)
```

---

## 6️⃣ Nadam

### 🔹 Concept

Adam + Nesterov momentum

```python
from tensorflow.keras.optimizers import Nadam(
    learning_rate=0.002
)
```

---

# 🔹 Optimizer Parameters (Common Summary)

| Parameter       | Meaning             |
| --------------- | ------------------- |
| `learning_rate` | Step size           |
| `momentum`      | Gradient memory     |
| `beta_1`        | First moment decay  |
| `beta_2`        | Second moment decay |
| `epsilon`       | Numerical stability |
| `weight_decay`  | Regularization      |

---

# 🔹 Optimizer vs Loss vs Metrics (Confusion Clear)

| Term      | Role               |
| --------- | ------------------ |
| Optimizer | Weight update      |
| Loss      | Error measure      |
| Metrics   | Performance report |

📌 **Optimizer uses loss, metrics does not affect training**

---

# 🔹 Learning Rate Scheduler (Advanced)

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96
)

optimizer = Adam(learning_rate=lr_schedule)
```

---

# 🔹 When to use which optimizer?

| Scenario           | Optimizer           |
| ------------------ | ------------------- |
| Beginner / general | Adam                |
| Fine control       | SGD                 |
| RNN                | RMSprop             |
| Sparse data        | Adagrad             |
| Transfer learning  | Adam / SGD (low LR) |

---

# ⚠️ Common Mistakes (VERY IMPORTANT)

❌ Learning rate খুব বড়
❌ Optimizer change করে recompile না করা
❌ Default Adam blindly সব জায়গায়
❌ Freeze layer কিন্তু optimizer change না করা

---

# 🧠 Interview-ready One-liners

* Optimizer controls how weights are updated
* Learning rate is the most critical hyperparameter
* Adam is adaptive and widely used
* Optimizer minimizes the loss function

---

# 📌 Minimal Working Example

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
```

---

## 🏁 Final Takeaway

> **Optimizer is the engine of learning — learning rate is its accelerator.**

---

