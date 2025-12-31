

---

# 📘 Metrics – Full Documentation (Accuracy, Precision, Recall)

---

## 🔹 1. Metrics কী?

**Metrics** হলো এমন measurement যেগুলো দিয়ে আমরা—

👉 Model কতটা ভালো prediction করছে
👉 Training / testing সময় performance কেমন

তা বুঝি।

📌 গুরুত্বপূর্ণ কথা:

* **Loss** → model শেখার জন্য (backpropagation)
* **Metrics** → model বিচার করার জন্য (human-readable)

---

## 🔹 2. `metrics` কোথায় ব্যবহার হয়?

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

📌 Metrics:

* Training output-এ দেখায়
* `model.evaluate()`-এ return হয়
* Training process **change করে না**

---

## 🔹 3. Confusion Matrix (Base Concept)

সব classification metric বোঝার জন্য এটা জানা বাধ্যতামূলক 👇

|                | Predicted YES       | Predicted NO        |
| -------------- | ------------------- | ------------------- |
| **Actual YES** | TP (True Positive)  | FN (False Negative) |
| **Actual NO**  | FP (False Positive) | TN (True Negative)  |

---

# 🔹 4. Accuracy

## 👉 Accuracy কী?

**Accuracy** বলে দেয়:

> মোট prediction-এর মধ্যে কয়টা ঠিক হয়েছে

---

### 📐 Formula

[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
]

---

### ✅ Keras

```python
metrics=['accuracy']
```

---

### 🔍 Example

* Total = 100
* Correct = 90

👉 Accuracy = **90%**

---

### ❌ Problem with Accuracy

Class imbalance থাকলে misleading হয়

Example:

* 95 negative
* 5 positive
* Model সব negative বলল

👉 Accuracy = 95% ❌ (কিন্তু useless)

---

# 🔹 5. Precision ⭐ (False Alarm Control)

## 👉 Precision কী?

**Precision** বলে দেয়:

> Model যেগুলোকে YES বলেছে, তার মধ্যে কয়টা সত্যি YES

---

### 📐 Formula

[
\text{Precision} = \frac{TP}{TP + FP}
]

---

### ✅ Keras

```python
metrics=['precision']
```

---

### 🔍 Example (Spam Detection)

| Case | Value |
| ---- | ----- |
| TP   | 40    |
| FP   | 10    |

👉 Precision = 40 / (40+10) = **0.80**

📌 মানে:

> ৮০% সময় model ঠিকভাবে spam ধরেছে

---

### 🧠 Precision কখন দরকার?

| Use Case        | Reason                       |
| --------------- | ---------------------------- |
| Spam detection  | False alarm কমাতে            |
| Email filter    | Good mail block না করতে      |
| Fraud detection | Innocent user accuse না করতে |

📌 **False Positive costly হলে → Precision গুরুত্বপূর্ণ**

---

# 🔹 6. Recall ⭐ (Missing Case Control)

## 👉 Recall কী?

**Recall** বলে দেয়:

> আসল YES গুলোর মধ্যে কয়টা model ধরতে পেরেছে

---

### 📐 Formula

[
\text{Recall} = \frac{TP}{TP + FN}
]

---

### ✅ Keras

```python
metrics=['recall']
```

---

### 🔍 Example (Disease Detection)

| Case | Value |
| ---- | ----- |
| TP   | 45    |
| FN   | 5     |

👉 Recall = 45 / (45+5) = **0.90**

📌 মানে:

> ৯০% রোগী detect হয়েছে

---

### 🧠 Recall কখন দরকার?

| Use Case          | Reason                   |
| ----------------- | ------------------------ |
| Disease detection | Patient miss করা যাবে না |
| Cancer screening  | False negative deadly    |
| Security threat   | Threat miss করা যাবে না  |

📌 **False Negative costly হলে → Recall গুরুত্বপূর্ণ**

---

# 🔹 7. Precision vs Recall (Most Important)

| Aspect         | Precision            | Recall                |
| -------------- | -------------------- | --------------------- |
| Focus          | False Positive       | False Negative        |
| Question       | “YES বললে কতটা ঠিক?” | “সব YES ধরতে পেরেছি?” |
| Important when | Innocent punish      | Real case miss        |

---

## 🧠 Easy Memory Trick ⭐

* **Precision** → *How precise my YES is*
* **Recall** → *How much I recalled from real YES*

---

# 🔹 8. Accuracy vs Precision vs Recall

| Metric    | Measures            | Problem              |
| --------- | ------------------- | -------------------- |
| Accuracy  | Overall correctness | Fails on imbalance   |
| Precision | False alarm         | Miss real cases      |
| Recall    | Missing cases       | Too many false alarm |

---

# 🔹 9. Keras-এ কীভাবে ব্যবহার হয়?

### ✅ Binary classification

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

---

### ✅ Evaluate return

```python
loss, acc, prec, rec = model.evaluate(x_test, y_test)
```

---

# 🔹 10. Common Mistakes ❌

### ❌ Accuracy-ই সব

```python
metrics=['accuracy']  # imbalance data
```

---

### ❌ Regression-এ precision/recall

```python
metrics=['precision']  # WRONG
```

---

## 🔹 11. Exam / Viva One-Liners ⭐

* **Accuracy overall correctness মাপে**
* **Precision false positive control করে**
* **Recall false negative control করে**
* **Medical domain → Recall important**
* **Spam/Fraud → Precision important**

---

# 🔹 12. Final Summary (Golden)

> 🔹 Loss model শেখায়
> 🔹 Accuracy মানুষকে বুঝায়
> 🔹 Precision বলে YES কতটা trustworthy
> 🔹 Recall বলে কতটা YES ধরা পড়েছে

---

