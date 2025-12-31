

---

# 📘 Keras / TensorFlow Core Function – Full Detailed Documentation

আমরা যেসব function cover করবো:

1. `load_data()`
2. `Sequential()`
3. `Dense()`
4. `Flatten()`
5. `model.compile()`
6. `model.fit()`
7. `model.evaluate()`
8. `model.predict()`
9. `model.summary()`

---

## 🔹 1. `mnist.load_data()`

### 📌 Purpose

👉 Dataset load করার জন্য
👉 MNIST digit data automatically download করে

---

### ✅ Syntax

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---

### 🔸 Parameters

```
None
```

📌 কোনো parameter লাগে না

---

### 🔁 Returns (খুব গুরুত্বপূর্ণ)

```
(x_train, y_train), (x_test, y_test)
```

| Variable | Type        | Shape           | Meaning         |
| -------- | ----------- | --------------- | --------------- |
| x_train  | numpy array | (60000, 28, 28) | Training images |
| y_train  | numpy array | (60000,)        | Training labels |
| x_test   | numpy array | (10000, 28, 28) | Test images     |
| y_test   | numpy array | (10000,)        | Test labels     |

---

### 📌 কোথায় ব্যবহার হবে?

* Training data হিসেবে → `fit()`
* Test data হিসেবে → `evaluate()`, `predict()`

---

## 🔹 2. `Sequential()`

### 📌 Purpose

👉 Neural Network model বানানোর জন্য
👉 Layers গুলো **একটার পর একটা** বসাতে

---

### ✅ Syntax

```python
model = Sequential(layers)
```

---

### 🔸 Parameters (Minimal)

| Parameter | Type | Meaning                              |
| --------- | ---- | ------------------------------------ |
| layers    | list | Dense / Flatten / Conv layer এর list |

---

### 🔁 Returns

```
model object
```

এই `model` object দিয়েই:

* `compile()`
* `fit()`
* `evaluate()`
* `predict()`
  সব কাজ হয়

---

### 📌 কোথায় ব্যবহার হবে?

* Simple model
* Single input → single output
* No branching

---

## 🔹 3. `Dense()`

### 📌 Purpose

👉 Fully Connected layer বানানোর জন্য
👉 Feature combine করে decision নেয়

---

### ✅ Syntax

```python
Dense(units, activation=None)
```

---

### 🔸 Parameters (Minimal + Important)

| Parameter  | Mandatory | Value                        |
| ---------- | --------- | ---------------------------- |
| units      | ✅         | Output neuron সংখ্যা         |
| activation | ❌         | 'relu', 'sigmoid', 'softmax' |

---

### 🔁 Returns

```
Dense layer object
```

---

### 📌 কোন ক্ষেত্রে কোন value?

| Case         | units          | activation    |
| ------------ | -------------- | ------------- |
| Regression   | 1              | linear / None |
| Binary class | 1              | sigmoid       |
| Multi-class  | no. of classes | softmax       |
| Hidden layer | any            | relu          |

---

## 🔹 4. `Flatten()`

### 📌 Purpose

👉 Image / multi-dimensional data → 1D vector
👉 Dense layer-এর আগে লাগবে

---

### ✅ Syntax

```python
Flatten()
```

---

### 🔸 Parameters

```
None
```

---

### 🔁 Returns

```
Flatten layer object
```

---

### 📌 কোথায় ব্যবহার হবে?

* Image data (28×28)
* CNN output → Dense

---

## 🔹 5. `model.compile()`

### 📌 Purpose

👉 Model-কে training-এর জন্য প্রস্তুত করা

---

### ✅ Syntax (Minimal)

```python
model.compile(optimizer, loss)
```

---

### 🔸 Parameters (Mandatory)

| Parameter | Meaning            | Example                           |
| --------- | ------------------ | --------------------------------- |
| optimizer | Weight update rule | 'adam'                            |
| loss      | Error calculation  | 'sparse_categorical_crossentropy' |

---

### 🔸 Optional (But common)

```python
metrics=['accuracy']
```

---

### 🔁 Returns

```
None
```

📌 কিন্তু internal state তৈরি হয়

---

### 📌 কোন ক্ষেত্রে কোন loss?

| Problem        | Loss                            |
| -------------- | ------------------------------- |
| Integer labels | sparse_categorical_crossentropy |
| One-hot labels | categorical_crossentropy        |
| Binary         | binary_crossentropy             |
| Regression     | mean_squared_error              |

---

## 🔹 6. `model.fit()`

### 📌 Purpose

👉 Model training শুরু করা

---

### ✅ Syntax (Minimal)

```python
model.fit(x, y, epochs)
```

---

### 🔸 Parameters

| Parameter | Mandatory | Meaning                 |
| --------- | --------- | ----------------------- |
| x         | ✅         | Training data           |
| y         | ✅         | Training labels         |
| epochs    | ✅         | কয়বার dataset train হবে |

---

### 🔁 Returns

```
History object
```

📌 History object দিয়ে:

```python
history.history['loss']
```

loss / accuracy plot করা যায়

---

## 🔹 7. `model.evaluate()`

### 📌 Purpose

👉 Trained model কতটা ভালো কাজ করছে সেটা মাপা

---

### ✅ Syntax

```python
model.evaluate(x, y)
```

---

### 🔸 Parameters

| Parameter | Meaning     |
| --------- | ----------- |
| x         | Test data   |
| y         | True labels |

---

### 🔁 Returns

```
loss, metrics
```

Example:

```python
loss, acc = model.evaluate(x_test, y_test)
```

---

## 🔹 8. `model.predict()`

### 📌 Purpose

👉 Model দিয়ে prediction বের করা

---

### ✅ Syntax

```python
model.predict(x)
```

---

### 🔸 Parameters

| Parameter | Meaning    |
| --------- | ---------- |
| x         | Input data |

---

### 🔁 Returns

| Task        | Return             |
| ----------- | ------------------ |
| Regression  | predicted value    |
| Binary      | probability        |
| Multi-class | probability vector |

Example:

```python
pred = model.predict(x_test)
argmax(pred[i]) → predicted class
```

---

## 🔹 9. `model.summary()`

### 📌 Purpose

👉 Model architecture দেখানো

---

### ✅ Syntax

```python
model.summary()
```

---

### 🔸 Parameters

```
None
```

---

### 🔁 Returns

```
None (prints table)
```

Shows:

* Layer name
* Output shape
* Parameter count

---

# 🧠 Master Summary Table ⭐

| Function   | Takes           | Returns      | Used For         |
| ---------- | --------------- | ------------ | ---------------- |
| load_data  | None            | data         | Dataset          |
| Sequential | layers          | model        | Model creation   |
| Dense      | units           | layer        | FC layer         |
| Flatten    | None            | layer        | Shape change     |
| compile    | optimizer, loss | None         | Prepare training |
| fit        | x, y, epochs    | History      | Training         |
| evaluate   | x, y            | loss, metric | Testing          |
| predict    | x               | prediction   | Inference        |
| summary    | None            | None         | Architecture     |

---

## ✅ Final Takeaway (Exam Line)

> **Deep Learning workflow:
> load → preprocess → build → compile → fit → evaluate → predict**

---

