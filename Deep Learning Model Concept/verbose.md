

---

# 🔊 `verbose` কী?

👉 **`verbose` ঠিক করে training চলাকালীন কীভাবে progress দেখাবে**।

📌 এটা **model-এর শেখায় কোনো প্রভাব ফেলে না**,
শুধু **display / logging** নিয়ন্ত্রণ করে।

---

# 🔹 `verbose`–এর possible values

| Value    | অর্থ                  |
| -------- | --------------------- |
| `0`      | কিছুই দেখাবে না       |
| `1`      | Progress bar দেখাবে   |
| `2`      | Epoch-wise simple log |
| `'auto'` | Environment অনুযায়ী   |

---

## 🔹 Example Setup (একই model)

```python
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=3,
    batch_size=32,
    verbose=?
)
```

এখন দেখি আলাদা `verbose` দিলে কী দেখায় 👇

---

# 1️⃣ `verbose = 0` → Silent mode

```python
verbose = 0
```

### Output:

```
(nothing printed)
```

👉 কোনো progress, loss, epoch—কিছুই দেখা যাবে না।

### কখন ব্যবহার করবে?

* Automated training
* Server / production
* Clean logs দরকার হলে

---

# 2️⃣ `verbose = 1` → Progress Bar (Most common)

```python
verbose = 1
```

### Output (Notebook style):

```
Epoch 1/3
32/32 [==============================] - 1s - loss: 0.245 - val_loss: 0.198
Epoch 2/3
32/32 [==============================] - 0s - loss: 0.112 - val_loss: 0.095
Epoch 3/3
32/32 [==============================] - 0s - loss: 0.058 - val_loss: 0.051
```

👉 Progress bar দেখায়
👉 Batch-by-batch update হয়

### কখন ব্যবহার করবে?

✔ Jupyter Notebook
✔ Interactive training
✔ Visual feedback দরকার হলে

---

# 3️⃣ `verbose = 2` → Epoch-wise clean log

```python
verbose = 2
```

### Output:

```
Epoch 1/3
 - loss: 0.245 - val_loss: 0.198
Epoch 2/3
 - loss: 0.112 - val_loss: 0.095
Epoch 3/3
 - loss: 0.058 - val_loss: 0.051
```

👉 কোনো progress bar নেই
👉 শুধু **epoch শেষে summary**

### কখন ব্যবহার করবে?

✔ Terminal / script
✔ Log file save
✔ Clean output দরকার হলে

---

# 4️⃣ `verbose = 'auto'` → Smart mode

```python
verbose = 'auto'
```

### Behavior:

* Notebook → `verbose=1`
* Script → `verbose=2`

📌 Default behaviour

---

# 🧠 Side-by-Side Comparison

| verbose  | Output style  | Best use            |
| -------- | ------------- | ------------------- |
| `0`      | No output     | Silent / production |
| `1`      | Progress bar  | Notebook            |
| `2`      | Epoch summary | Script / logging    |
| `'auto'` | Smart         | Default             |

---

# 🔹 Important Note

`verbose`:

* ❌ training speed বদলায় না
* ❌ accuracy/loss বদলায় না
* ✅ শুধু output দেখায়

---

# 🧪 Mini Code Demo (Try Yourself)

```python
for v in [0, 1, 2]:
    print(f"\nVerbose = {v}")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=2,
        verbose=v
    )
```

---

# 🧠 Interview One-liner

> `verbose` controls how training progress is displayed, without affecting the learning process.

---

# ✅ Final Takeaway

✔ Notebook → `verbose=1`
✔ Script → `verbose=2`
✔ Production → `verbose=0`

---

