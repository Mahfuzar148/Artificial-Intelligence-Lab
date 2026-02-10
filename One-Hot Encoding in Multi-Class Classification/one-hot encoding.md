# Topic: One-Hot Encoding in Multi-Class Classification (MNIST Dataset)

---

## 1. Introduction

In multi-class classification problems like **MNIST handwritten digit recognition**, the model must predict one class out of multiple possible classes (digits 0–9). To make this work correctly with neural networks, especially when using **Softmax activation** and **Categorical Crossentropy loss**, labels must be converted into a special format called **one-hot encoding**.

This document explains:

* What the original label format is
* Why it is not sufficient
* What one-hot encoding does
* How label shape changes from `(60000,)` to `(60000, 10)`

---

## 2. Original Label Format (Before One-Hot Encoding)

After loading the MNIST dataset:

```python
(trainX, trainY), (testX, testY) = load_data()
```

### 2.1 Shape of Labels

```python
trainY.shape = (60000,)
```

### 2.2 What This Means

* There are **60000 training images**
* Each image has **one integer label**
* Label value ranges from **0 to 9**

### 2.3 Example

```python
trainY = [5, 0, 4, 1, 9, ...]
```

This means:

* First image is digit **5**
* Second image is digit **0**
* Third image is digit **4**

This format is called **integer-encoded labels**.

---

## 3. Output Layer Requirement in Neural Networks

In a typical MNIST classification model, the output layer is:

```python
Dense(10, activation='softmax')
```

### 3.1 Why 10 Neurons?

* There are **10 possible classes** (digits 0–9)
* Each neuron represents the probability of one digit

### 3.2 Softmax Output Example

```text
[0.01, 0.02, 0.05, 0.03, 0.10, 0.70, 0.04, 0.02, 0.02, 0.01]
```

This means:

* Probability of digit 5 is highest (0.70)

---

## 4. Problem with Integer Labels

If the true label is:

```python
trainY[0] = 5
```

The model cannot directly compare:

```text
5  ❌  vs  [0.01, 0.02, ..., 0.70, ...]
```

Because:

* Output is a **vector of probabilities**
* Label is a **single number**

So loss calculation becomes ambiguous.

---

## 5. What Is One-Hot Encoding?

One-hot encoding converts an integer label into a **binary vector**.

### 5.1 Example Conversion

```python
to_categorical(5, num_classes=10)
```

Output:

```text
[0 0 0 0 0 1 0 0 0 0]
```

### 5.2 Meaning

* Vector length = number of classes (10)
* Index 5 is set to 1
* All other indices are 0

This clearly tells the model:

> “This image belongs to class 5.”

---

## 6. Applying One-Hot Encoding to All Labels

```python
trainY = to_categorical(trainY, num_classes=10)
testY  = to_categorical(testY, num_classes=10)
```

### 6.1 Shape Change

Before encoding:

```text
trainY.shape = (60000,)
```

After encoding:

```text
trainY.shape = (60000, 10)
```

---

## 7. How to Interpret `(60000, 10)`

This shape means:

* **60000 rows** → one row per image
* **10 columns** → one column per class (0–9)

### 7.1 Row-wise Representation

| Image | One-Hot Label         |
| ----- | --------------------- |
| img₁  | [0 0 0 0 0 1 0 0 0 0] |
| img₂  | [1 0 0 0 0 0 0 0 0 0] |
| img₃  | [0 0 0 0 1 0 0 0 0 0] |

---

## 8. Why One-Hot Encoding Is Necessary

One-hot encoding is required because:

* Softmax outputs probabilities for **each class**
* Categorical Crossentropy compares **two vectors**
* Vector-to-vector comparison ensures correct loss calculation

Without one-hot encoding, training will fail or give incorrect results.

---

## 9. Alternative (Important Note)

If labels are kept as integers `(60000,)`, then loss function must be:

```python
loss = 'sparse_categorical_crossentropy'
```

But when using:

```python
loss = 'categorical_crossentropy'
```

👉 **One-hot encoding is mandatory**.

---

## 10. Conclusion

* Original labels are integers with shape `(60000,)`
* Neural network output is a probability vector of length 10
* One-hot encoding converts each label into a 10-length binary vector
* This changes label shape to `(60000, 10)`
* One-hot encoding ensures correct learning and loss calculation in multi-class classification

---

### End of Document
