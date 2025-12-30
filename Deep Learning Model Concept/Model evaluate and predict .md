

---

# 🧾 `model.evaluate()` — Full Documentation

## 🔹 `model.evaluate()` কী?

👉 `model.evaluate()` ব্যবহার করা হয় **trained model-এর performance মাপার জন্য**
👉 সাধারণত **test data**–তে চালানো হয়

📌 এটি **training করে না**, শুধু **measurement** করে।

---

## 🔹 Basic Syntax

```python
model.evaluate(
    x,
    y=None,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    return_dict=False
)
```

---

## 🔴 Mandatory Parameters

### 1️⃣ `x` ✅

```python
x_test
```

👉 Input data (features)

Accepts:

* NumPy array
* Tensor
* list / dict (multi-input)

---

### 2️⃣ `y` ✅ (Supervised learning)

```python
y_test
```

👉 True labels / ground truth

❌ না দিলে → loss/metric calculate করা যাবে না

---

## 🟡 Core Optional Parameters

### 3️⃣ `batch_size`

```python
batch_size=32
```

👉 একবারে কত sample নিয়ে evaluation হবে

| Behaviour             |
| --------------------- |
| ছোট batch → কম memory |
| বড় batch → দ্রুত      |

📌 Default = training batch size

---

### 4️⃣ `verbose`

```python
verbose=0
```

| Value | Output           |
| ----- | ---------------- |
| `0`   | কোনো output না   |
| `1`   | Progress bar     |
| `2`   | Line-wise output |

📌 Test evaluation সাধারণত silent রাখা হয়

---

### 5️⃣ `steps`

👉 Generator / `tf.data` ব্যবহার করলে লাগে

```python
steps = number_of_batches
```

---

### 6️⃣ `return_dict`

```python
return_dict=True
```

👉 Output dictionary আকারে দিবে

Example:

```python
{'loss': 0.0012, 'mae': 0.02}
```

---

## 🔹 Output of `model.evaluate()`

Return করে:

```python
loss, metric1, metric2, ...
```

Order ঠিক থাকে যেভাবে compile এ দিয়েছো

---

## 🔍 তোমার Code Explained

```python
test_loss, test_mae = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
```

এর মানে:

* `x_test, y_test` → unseen data
* `verbose=0` → silent evaluation
* `test_loss` → loss function value (MSE)
* `test_mae` → metric value (MAE)

---

# 🧾 `model.predict()` — Full Documentation

## 🔹 `model.predict()` কী?

👉 `model.predict()` ব্যবহার করা হয়—

> **model কী output দিচ্ছে তা বের করার জন্য**

📌 এখানে:

* loss লাগে না
* label লাগে না
* weight update হয় না

---

## 🔹 Basic Syntax

```python
model.predict(
    x,
    batch_size=None,
    verbose='auto',
    steps=None
)
```

---

## 🔴 Mandatory Parameter

### 1️⃣ `x` ✅

```python
x_test
```

👉 Input features

---

## 🟡 Optional Parameters

### 2️⃣ `batch_size`

```python
batch_size=32
```

👉 Prediction speed & memory control

---

### 3️⃣ `verbose`

```python
verbose=0
```

👉 Prediction log দেখাবে কিনা

---

### 4️⃣ `steps`

👉 Generator-based prediction এর জন্য

---

## 🔹 Output of `model.predict()`

Return করে:

```python
y_pred
```

Shape:

```
(samples, output_units)
```

---

## 🔍 তোমার Code Explained

```python
y_pred_scaled = model.predict(x_test)
```

👉 Output এখনো **scaled form**–এ আছে
কারণ model scaled data দিয়ে train হয়েছে

---

# 🔄 Rescaling Explained (VERY IMPORTANT)

```python
y_pred = y_pred_scaled * max_y
y_true = y_test * max_y
```

### কেন দরকার?

কারণ training-এর সময়:

```python
y_scaled = y / max_y
```

Model শেখে scaled output

👉 আসল unit-এ ফেরাতে হলে:

```python
original = scaled × max_y
```

---

## 🔹 Without rescaling কী সমস্যা?

| Without rescale     | With rescale    |
| ------------------- | --------------- |
| Value ভুল unit      | Real-world unit |
| Interpretation কঠিন | Meaningful      |
| Plot confusing      | Correct plot    |

---

# 🔍 `evaluate()` vs `predict()` (Difference)

| Feature       | evaluate          | predict         |
| ------------- | ----------------- | --------------- |
| Purpose       | Performance check | Output generate |
| Needs labels  | ✅ Yes             | ❌ No            |
| Returns       | Loss + metrics    | Predictions     |
| Weight update | ❌ No              | ❌ No            |
| Use-case      | Test accuracy     | Inference       |

---

# 🧠 Typical ML Workflow

```text
compile()
fit()
evaluate()
predict()
```

---

# ⚠️ Common Mistakes

❌ Train data দিয়ে evaluate
❌ Scale mismatch
❌ Test data দিয়ে tune
❌ Predict করার পর inverse scale না করা

---

# 🧠 Interview-ready One-liners

* `evaluate()` measures model performance on labeled data
* `predict()` generates model outputs without labels
* Evaluation does not update weights
* Rescaling restores real-world units

---

# 📌 Summary Table

| Function | Goal                | Input | Output        |
| -------- | ------------------- | ----- | ------------- |
| evaluate | Measure performance | x + y | loss, metrics |
| predict  | Generate output     | x     | predictions   |

---

## 🏁 Final Takeaway

> **`evaluate()` tells how good the model is, `predict()` tells what the model predicts.**

---

