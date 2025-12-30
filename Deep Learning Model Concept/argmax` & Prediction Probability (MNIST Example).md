

---

# 📘 DOCUMENTATION: `argmax` & Prediction Probability (MNIST Example)

---

## 🔹 1. Model Prediction আসলে কী?

যখন তুমি লিখো:

```python
y_pred_prob = model.predict(x_test)
```

তখন model **final decision দেয় না**, বরং দেয় —

👉 **প্রতিটা class-এর probability**

### MNIST (10 digits) হলে:

একটা image এর জন্য output হয়:

```python
[0.01, 0.00, 0.02, 0.01, 0.00, 0.03, 0.00, 0.90, 0.02, 0.01]
```

এটার মানে:

* Digit 0 → 1%
* Digit 7 → **90% (সবচেয়ে বেশি)**
* সব probability এর যোগফল = 1

---

## 🔹 2. `argmax` কী? (Core Concept)

### Definition

> **`argmax` array-এর মধ্যে যেই element সবচেয়ে বড়, তার index বের করে**

### Simple example

```python
a = [5, 20, 3]
np.argmax(a)   # output: 1
```

কারণ:

* max value = 20
* index = 1

---

## 🔹 3. Classification এ `argmax` কেন দরকার?

Model output:

```python
y_pred_prob.shape = (num_samples, num_classes)
```

আমাদের দরকার:

```python
[7, 0, 4, 1, ...]   # final predicted digit
```

তাই আমরা লিখি:

```python
y_pred = np.argmax(y_pred_prob, axis=1)
```

---

## 🔹 4. `axis=1` মানে কী?

| axis     | অর্থ                      |
| -------- | ------------------------- |
| `axis=0` | column-wise               |
| `axis=1` | **row-wise (একটা image)** |

👉 যেহেতু:

* 1 row = 1 image
* প্রতিটা image থেকে max probability বের করতে চাই

✔ তাই `axis=1`

---

## 🔹 5. Full Flow (Prediction Logic)

```text
Image
 ↓
Softmax layer
 ↓
Probability vector (10 values)
 ↓
argmax
 ↓
Final digit prediction
```

---

## 🔹 6. Image + Probability একসাথে কিভাবে Print করবে

### 🎯 Goal:

একটা digit image দেখাবে
সাথে দেখাবে:

* True label
* Predicted label
* প্রতিটা digit-এর probability

---

## 🔹 Step 1: Prediction নাও

```python
y_pred_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

y_true = np.argmax(y_test, axis=1)
```

---

## 🔹 Step 2: একটি image বেছে নাও

```python
idx = 0   # যেকোনো index
```

---

## 🔹 Step 3: Image + Probability Print

```python
plt.figure(figsize=(12,4))

# 🔹 Image
plt.subplot(1,2,1)
plt.imshow(x_test[idx], cmap='gray')
plt.title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}")
plt.axis('off')

# 🔹 Probability bar chart
plt.subplot(1,2,2)
plt.bar(range(10), y_pred_prob[idx])
plt.xlabel("Digit")
plt.ylabel("Probability")
plt.title("Prediction Probabilities")
plt.show()
```

👉 এতে তুমি **চোখে দেখবে model কতটা confident**।

---

## 🖼️ Visual Idea (MNIST digits)

![Image](https://www.researchgate.net/publication/382145539/figure/fig4/AS%3A11431281259781353%401720667202742/Please-zoom-in-for-detail-Average-Softmax-Probabilities-for-Correctly-and-Incorrectly.ppm)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1296/1%2AXW3q3RmROtKbJSK13yHccg.jpeg)

(তোমার output এ এরকম image + bar chart আসবে)

---

## 🔹 7. Multiple Image + Probability (Bonus)

```python
plt.figure(figsize=(15,6))

for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"T:{y_true[i]} P:{y_pred[i]}")
    plt.axis('off')

    plt.subplot(2,5,i+6)
    plt.bar(range(10), y_pred_prob[i])
    plt.xticks(range(10))
```

---

## 🔹 8. Common Mistakes 🚨

❌ `argmax` না ব্যবহার করে accuracy বের করা
❌ `axis=0` ব্যবহার করা
❌ probability vector কে final label ধরা

---

## 🔑 One-Line Summary

> **Softmax probability বলে “কতটা বিশ্বাস”, `argmax` বলে “final decision”**

---

