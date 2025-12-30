
---

# 🧾 `model.summary()` — Full Documentation (Keras)

## 🔹 `model.summary()` কী?

`model.summary()` হলো Keras-এর একটি **inspection utility**—
এটা **model-এর architecture এক নজরে দেখায়**:

* কোন কোন layer আছে
* প্রতিটা layer-এর output shape
* কত parameter আছে
* কোনগুলো **trainable** আর কোনগুলো **non-trainable**

📌 **Training করে না**, শুধু **report প্রিন্ট করে**।

---

## 🔹 Basic Syntax

```python
model.summary()
```

---

## 🔹 Full Syntax (সব parameter)

```python
model.summary(
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False
)
```

> ⚠️ কিছু parameter **TensorFlow version অনুযায়ী** আসতে পারে/না-ও আসতে পারে।

---

# 1️⃣ `line_length`

### 🔹 কী কাজ করে?

এক লাইনে **কত character** দেখাবে—column width নিয়ন্ত্রণ করে।

```python
model.summary(line_length=120)
```

### 🔹 কখন দরকার?

* বড় model
* layer name / shape কাটছাঁট হয়ে গেলে

### 🔹 না নিলে?

* Default width ব্যবহার হবে (সাধারণত 80)

---

# 2️⃣ `positions`

### 🔹 কী কাজ করে?

Column গুলো **কোথায় বসবে** সেটা manual ভাবে নির্ধারণ করে।

```python
model.summary(positions=[0.3, 0.6, 0.75, 1.0])
```

### 🔹 Column কী কী?

* Layer (type)
* Output Shape
* Param #

### 🔹 না নিলে?

* Keras নিজে auto-layout করে

📌 **Advanced formatting**, সাধারণত লাগে না।

---

# 3️⃣ `print_fn`

### 🔹 কী কাজ করে?

Summary কোথায় print হবে তা নির্ধারণ করে।

```python
model.summary(print_fn=lambda x: my_list.append(x))
```

### 🔹 Use-case

* File-এ save করতে
* Logger-এ পাঠাতে

**Example: file-এ save**

```python
with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
```

### 🔹 না নিলে?

* `stdout` (console) এ print হবে

---

# 4️⃣ `expand_nested`

### 🔹 কী কাজ করে?

Nested model (model-এর ভিতরে model) **খুলে দেখাবে কিনা**।

```python
model.summary(expand_nested=True)
```

### 🔹 Use-case

* Transfer learning
* Pretrained model (VGG, ResNet) ভিতরের layer দেখতে

### 🔹 না নিলে?

* Nested model এক লাইনে collapse হয়ে থাকবে

---

# 5️⃣ `show_trainable` (VERY IMPORTANT)

### 🔹 কী কাজ করে?

প্রতিটা layer-এর পাশে **Trainable=True/False** দেখায়।

```python
model.summary(show_trainable=True)
```

### 🔹 Output Example

```
dense_1 (Dense)  (None, 8)  16  Trainable=True
```

### 🔹 কেন দরকার?

* Weight freeze verify করতে
* Transfer learning debugging

### 🔹 না নিলে?

* Trainable info শুধু **bottom summary** তে থাকবে

---

# 🔍 `model.summary()` Output কীভাবে পড়বে?

### Typical Output

```text
Layer (type)            Output Shape        Param #
===================================================
input_1 (InputLayer)   [(None, 1)]         0
hidden1 (Dense)        (None, 8)            16
hidden2 (Dense)        (None, 16)           144
output (Dense)         (None, 1)            17
===================================================
Total params: 177
Trainable params: 177
Non-trainable params: 0
```

---

## 🔹 Column-by-column ব্যাখ্যা

### 1️⃣ Layer (type)

* Layer নাম + class
* Debugging সহজ

### 2️⃣ Output Shape

* `(None, units)`
* `None` = batch size (runtime-এ আসবে)

### 3️⃣ Param

* ঐ layer-এর মোট parameter সংখ্যা

---

## 🔹 Bottom Lines (সবচেয়ে গুরুত্বপূর্ণ)

```text
Trainable params: 177
Non-trainable params: 0
```

* **Trainable** → backprop এ update হবে
* **Non-trainable** → freeze করা weight

---

# 🧮 Parameter Count কিভাবে আসে? (Dense)

```text
Params = (input_features × units) + units
```

**Example**

```python
Dense(8) with input (1,)
→ (1×8) + 8 = 16
```

---

# 🧊 Weight Freeze হলে Summary কেমন হয়?

```python
for layer in model.layers:
    layer.trainable = False

model.compile(...)
model.summary()
```

Output:

```text
Trainable params: 0
Non-trainable params: 177
```

---

# 🔍 Layer-wise Trainable Check (Complementary)

```python
for layer in model.layers:
    print(layer.name, layer.trainable, layer.count_params())
```

---

# ⚠️ Common Mistakes

❌ Freeze করার পর compile না করা
❌ Summary দেখে “trainable” বোঝা না
❌ Nested model খুলে না দেখা

---

# ✅ Best Practices

* Freeze/unfreeze করার **পর** `model.summary()` দেখো
* Transfer learning এ `show_trainable=True` ব্যবহার করো
* বড় model এ `expand_nested=True`

---

# 🧠 Interview-ready One-liners

* **`model.summary()`** shows architecture and parameter counts
* **Trainable params** are updated during backprop
* **Non-trainable params** are frozen weights

---

# 📌 Quick Reference

| Parameter        | Mandatory | কাজ                  |
| ---------------- | --------- | -------------------- |
| `line_length`    | ❌         | Column width         |
| `positions`      | ❌         | Column position      |
| `print_fn`       | ❌         | Custom print         |
| `expand_nested`  | ❌         | Nested model expand  |
| `show_trainable` | ❌         | Layer-wise trainable |

---

## 🔑 Final Takeaway

> **`model.summary()` হলো model debugging-এর সবচেয়ে শক্তিশালী টুল**—
> architecture, parameters, আর trainability এক নজরে বোঝা যায়।

