

তোমার reference code 👇

```python
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)
```

---

# 🧾 `model.fit()` — Full Documentation (Keras)

## 🔹 `model.fit()` কী?

👉 `model.fit()` হলো **training engine**।
এখানেই model:

* data দেখে
* loss হিসাব করে
* gradient বের করে
* optimizer দিয়ে weight update করে

📌 **compile() না করলে fit() চলবে না**

---

## 🔹 Full Syntax (All Parameters)

```python
model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose='auto',
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None
)
```

---

# 🔴 Mandatory Parameters

## 1️⃣ `x` (REQUIRED)

```python
x_train
```

### 🔹 কাজ

👉 Model-এর **input data**

### 🔹 Accepts

* NumPy array
* Tensor
* list / dict (multi-input model)

### ❌ না দিলে

Error আসবে

---

## 2️⃣ `y` (REQUIRED)

```python
y_train
```

### 🔹 কাজ

👉 Ground truth / target / label

📌 Supervised learning-এ mandatory

---

# 🟡 Core Training Parameters

---

## 3️⃣ `epochs`

```python
epochs=50
```

### 🔹 কাজ

👉 **পুরো training dataset কতবার model দেখবে**

### 🔹 Behaviour

| Epoch value     | Effect       |
| --------------- | ------------ |
| ছোট (10)        | Underfitting |
| মাঝারি (50–100) | Balanced     |
| খুব বড় (500)    | Overfitting  |

📌 EarlyStopping থাকলে epoch বড় দিলেও সমস্যা নেই

---

## 4️⃣ `batch_size`

```python
batch_size=32
```

### 🔹 কাজ

👉 একবারে কত sample নিয়ে gradient update হবে

### 🔹 Behaviour

| Batch size  | Effect        |
| ----------- | ------------- |
| ছোট (8, 16) | Stable, slow  |
| মাঝারি (32) | Best tradeoff |
| বড় (128+)   | Fast, noisy   |

📌 Default = 32

---

## 5️⃣ `validation_data`

```python
validation_data=(x_val, y_val)
```

### 🔹 কাজ

👉 Model training-এর মাঝে **নিজেকে যাচাই করবে**

📌 Validation data দিয়ে weight update হয় না

### 🔹 কেন দরকার?

* Overfitting detect
* Hyperparameter tuning
* EarlyStopping trigger

---

### ❌ না দিলে?

* `val_loss`, `val_accuracy` থাকবে না
* EarlyStopping কাজ করবে না

---

## 🔁 Alternative: `validation_split`

```python
validation_split=0.2
```

👉 Training data থেকেই 20% validation বানাবে

⚠️ `validation_data` আর `validation_split` একসাথে দেওয়া যায় না

---

# 🔵 Callback Parameters

---

## 6️⃣ `callbacks`

```python
callbacks=[early_stop]
```

### 🔹 কাজ

👉 Training চলাকালীন **extra control**

### 🔹 Common callbacks

* `EarlyStopping`
* `ModelCheckpoint`
* `ReduceLROnPlateau`
* `TensorBoard`

### 🔹 Behaviour

* EarlyStopping → training আগে থামাবে
* ModelCheckpoint → best model save করবে

---

## 🧾 `history` object কী?

```python
history = model.fit(...)
```

👉 Training log store করে

### 🔹 Access

```python
history.history.keys()
```

Output:

```text
['loss', 'val_loss', 'mae', 'val_mae']
```

---

# 🔈 Output Control Parameters

---

## 7️⃣ `verbose`

```python
verbose=2
```

### 🔹 কাজ

👉 Training log কিভাবে দেখাবে

### 🔹 Values

| Value    | Output          |
| -------- | --------------- |
| `0`      | কিছুই দেখাবে না |
| `1`      | Progress bar    |
| `2`      | Epoch-wise line |
| `'auto'` | Default         |

📌 Notebook-এ `verbose=1` সুন্দর
📌 Script-এ `verbose=2` পরিষ্কার

---

# 🟢 Optional but Important Parameters

---

## 8️⃣ `shuffle`

```python
shuffle=True
```

### 🔹 কাজ

👉 প্রতিটা epoch-এর আগে data shuffle করবে

### 🔹 Default

✔ True

❌ Time-series data হলে False দিতে হবে

---

## 9️⃣ `steps_per_epoch`

### 🔹 কাজ

👉 এক epoch-এ কত batch চলবে

```python
steps_per_epoch = total_samples // batch_size
```

📌 Generator / tf.data ব্যবহার করলে দরকার

---

## 🔟 `initial_epoch`

```python
initial_epoch=10
```

### 🔹 কাজ

👉 Training resume করার জন্য

---

# 🔍 তোমার Code-এর Behaviour (Line by Line)

```python
history = model.fit(
    x_train, y_train,             # training data
    validation_data=(x_val, y_val),# validation check
    epochs=50,                    # max 50 epoch
    batch_size=32,                # 32 sample per update
    callbacks=[early_stop],       # early stopping enabled
    verbose=2                     # epoch-wise log
)
```

### এর মানে:

* Model সর্বোচ্চ 50 epoch চলবে
* কিন্তু `early_stop` চাইলে আগে থামাতে পারে
* প্রতি epoch শেষে validation loss চেক হবে
* Best weight restore হবে

---

# ⚠️ Common Mistakes (VERY IMPORTANT)

❌ validation_data না দিয়ে EarlyStopping
❌ batch_size খুব বড় দেওয়া
❌ epochs কম দেওয়া
❌ history object ignore করা

---

# 🧠 Interview-ready One-liners

* `epochs` = full data passes
* `batch_size` = samples per update
* `validation_data` = overfitting check
* `callbacks` = training control hooks
* `history` = training log

---

# 📌 Summary Table

| Parameter       | Mandatory | কাজ             |
| --------------- | --------- | --------------- |
| x, y            | ✅         | Training data   |
| epochs          | ❌         | Training length |
| batch_size      | ❌         | Update size     |
| validation_data | ❌         | Validation      |
| callbacks       | ❌         | Control         |
| verbose         | ❌         | Output          |

---

## 🏁 Final Takeaway

> **`model.fit()` defines how, how long, and under what control a model learns.**

