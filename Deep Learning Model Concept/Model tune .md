
---

# 🧾 Model Tuning (Hyperparameter Tuning) — Full Documentation

---

## 🔹 “Tune করা” মানে কী?

👉 **Model tuning** মানে হলো:

> model-এর এমন সব setting (hyperparameters) পরিবর্তন করা
> যেগুলো model নিজে শেখে না,
> কিন্তু শেখার quality-কে সরাসরি প্রভাব ফেলে।

📌 সহজ ভাষায়:
**“Model কীভাবে শিখবে, সেটা ঠিক করা = tuning”**

---

## 🔹 Train বনাম Tune (Difference Clear)

| বিষয়       | Train         | Tune               |
| ---------- | ------------- | ------------------ |
| কী শেখা হয় | Weights, bias | Hyperparameters    |
| কে শেখে    | Model নিজে    | আমরা (developer)   |
| কোন data   | Train data    | Validation data    |
| Goal       | Loss কমানো    | Best configuration |

---

## 🔹 Hyperparameter কী?

👉 Hyperparameter হলো এমন parameter—

* training শুরু হওয়ার **আগে সেট করা হয়**
* training চলাকালীন **update হয় না**

---

## 🔹 Common Hyperparameters (Most Important List)

1️⃣ Learning Rate
2️⃣ Number of Epochs
3️⃣ Batch Size
4️⃣ Number of Layers
5️⃣ Number of Neurons
6️⃣ Activation Function
7️⃣ Optimizer Type
8️⃣ Regularization (L2, Dropout)
9️⃣ EarlyStopping settings

---

# 🔷 1️⃣ Learning Rate (সবচেয়ে গুরুত্বপূর্ণ)

### 🔹 কী?

👉 Weight update কত বড় step-এ হবে

```python
optimizer = Adam(learning_rate=0.001)
```

### 🔹 Tune না করলে কী সমস্যা?

| LR      | Problem           |
| ------- | ----------------- |
| খুব বড়  | Loss diverge      |
| খুব ছোট | Training খুব slow |
| ঠিক     | Smooth learning   |

### 🔹 Tuning example

```python
LR = 0.01  → val_loss = 0.45 ❌
LR = 0.001 → val_loss = 0.18 ✅
```

---

# 🔷 2️⃣ Epochs

### 🔹 কী?

👉 পুরো dataset কতবার model দেখবে

```python
epochs = 200
```

### 🔹 Tune না করলে?

* কম epoch → underfitting
* বেশি epoch → overfitting

### 🔹 Tuning example

```python
epochs = 50  → val_loss = 0.30
epochs = 120 → val_loss = 0.20 ✅
```

---

# 🔷 3️⃣ Batch Size

### 🔹 কী?

👉 একবারে কত sample দিয়ে weight update হবে

```python
batch_size = 32
```

### 🔹 Effect

| Batch Size | Result       |
| ---------- | ------------ |
| ছোট        | Stable, slow |
| বড়         | Fast, noisy  |

### 🔹 Tuning example

```python
batch=16 → val_acc=87%
batch=32 → val_acc=90% ✅
```

---

# 🔷 4️⃣ Number of Layers

### 🔹 কী?

👉 Model কতটা deep হবে

```python
Dense → Dense → Dense
```

### 🔹 Tune না করলে?

* কম layer → underfitting
* বেশি layer → overfitting

---

# 🔷 5️⃣ Number of Neurons

### 🔹 কী?

👉 প্রতিটা layer কতটা capacity রাখবে

```python
Dense(8)
Dense(32)
```

### 🔹 Tuning example

```python
Dense(4)  → val_loss = 0.40 ❌
Dense(16) → val_loss = 0.19 ✅
```

---

# 🔷 6️⃣ Activation Function

### 🔹 কী?

👉 Non-linearity যোগ করে

| Activation | Use               |
| ---------- | ----------------- |
| ReLU       | Hidden layer      |
| Sigmoid    | Binary output     |
| Softmax    | Multi-class       |
| Linear     | Regression output |

### 🔹 Wrong activation দিলে?

❌ Model learn করতে পারবে না

---

# 🔷 7️⃣ Optimizer

### 🔹 কী?

👉 Weight update করার algorithm

| Optimizer | Use                 |
| --------- | ------------------- |
| Adam      | Default / fast      |
| SGD       | Controlled learning |
| RMSprop   | Sequence data       |

### 🔹 Tuning example

```python
SGD  → slow convergence
Adam → fast convergence ✅
```

---

# 🔷 8️⃣ Regularization (Overfitting Control)

## 🔸 Dropout

```python
Dropout(0.3)
```

👉 30% neuron randomly বন্ধ

## 🔸 L2 Regularization

```python
Dense(16, kernel_regularizer=l2(0.001))
```

### 🔹 Tune না করলে?

* Overfitting হবে

---

# 🔷 9️⃣ EarlyStopping (Auto Tuning)

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

👉 Model নিজেই বলে দেয়:

> “আর improve হচ্ছে না”

---

# 🔹 Tune করার সময় কোন data ব্যবহার হবে?

| Data       | Purpose          |
| ---------- | ---------------- |
| Train      | Weights শেখা     |
| Validation | 🔧 Tuning        |
| Test       | Final evaluation |

❌ Test data দিয়ে tuning করা **strictly forbidden**

---

# 🔄 Manual Tuning Process (Step-by-Step)

```text
1. Train model
2. Check validation metric
3. Change ONE hyperparameter
4. Train again
5. Compare results
6. Keep best configuration
```

---

# 🧪 Mini End-to-End Tuning Example

```python
# Try LR = 0.01
model.compile(optimizer=Adam(0.01), loss='mse')
h1 = model.fit(...)

# Try LR = 0.001
model.compile(optimizer=Adam(0.001), loss='mse')
h2 = model.fit(...)
```

```python
min(h1.history['val_loss']),
min(h2.history['val_loss'])
```

---

# ⚠️ Common Mistakes (VERY IMPORTANT)

❌ Test data দিয়ে tune করা
❌ একসাথে অনেক hyperparameter change করা
❌ EarlyStopping ব্যবহার না করা
❌ Best model save না করা

---

# 🧠 Best Practices (Industry)

✔ One parameter at a time
✔ Validation-based decision
✔ EarlyStopping + ModelCheckpoint
✔ Log everything
✔ Fix random seed

---

# 🧠 Interview One-liners

* Tuning adjusts hyperparameters, not weights
* Validation data is used for tuning
* Learning rate is the most critical hyperparameter
* EarlyStopping is an automatic tuning method

---

# 📌 Final Summary Table

| Aspect         | Meaning            |
| -------------- | ------------------ |
| Train          | Learn weights      |
| Tune           | Adjust settings    |
| Hyperparameter | Predefined control |
| Validation     | Tuning data        |
| Test           | Final check        |

---

## 🏁 Golden Rule

> **Train on train set, tune on validation set, evaluate on test set.**

---

