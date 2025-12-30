

---

# 📘 DATASET LOADING — FULL DOCUMENTATION (TensorFlow / Keras)

---

## 1️⃣ Keras Built-in Dataset (সবচেয়ে সহজ)

TensorFlow কিছু dataset আগে থেকেই দিয়ে রাখে।

### 📂 Available datasets

| Dataset        | Problem                 |
| -------------- | ----------------------- |
| MNIST          | Handwritten digits      |
| Fashion-MNIST  | Clothing classification |
| CIFAR-10       | 10 object classes       |
| CIFAR-100      | 100 object classes      |
| IMDB           | Text sentiment          |
| Reuters        | News classification     |
| Boston Housing | Regression              |

---

### 🔹 Import rule (সবগুলোর জন্য same)

```python
import tensorflow as tf
```

---

### 🔹 General syntax

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.DATASET_NAME.load_data()
```

---

### 🔹 Examples

#### MNIST

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

#### Fashion-MNIST

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

#### CIFAR-10 (RGB image)

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

---

### 🔹 Shape difference

| Dataset       | x_train shape      |
| ------------- | ------------------ |
| MNIST         | (60000, 28, 28)    |
| Fashion-MNIST | (60000, 28, 28)    |
| CIFAR-10      | (50000, 32, 32, 3) |

---

## 2️⃣ CSV / Tabular Dataset Load

### 🔹 Using Pandas

```python
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop("label", axis=1)
y = df["label"]
```

---

### 🔹 Convert to NumPy

```python
X = X.values
y = y.values
```

---

## 3️⃣ Image Dataset (Folder Structure)

### 📁 Folder format (mandatory)

```
dataset/
 ├── train/
 │   ├── cat/
 │   └── dog/
 └── test/
     ├── cat/
     └── dog/
```

---

### 🔹 Load using Keras

```python
from tensorflow.keras.utils import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)

test_ds = image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32
)
```

---

## 4️⃣ Text Dataset Load

### 🔹 IMDB Sentiment

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
```

👉 Output: integer-encoded text

---

### 🔹 Custom text file

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
```

---

## 5️⃣ TensorFlow Dataset API (`tf.data`)

### 🔹 Why use it?

✔ Faster
✔ Handles big data
✔ Pipeline optimization

---

### 🔹 From NumPy

```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1000).batch(32)
```

---

### 🔹 From CSV

```python
dataset = tf.data.experimental.make_csv_dataset(
    "data.csv",
    batch_size=32,
    label_name="label"
)
```

---

## 6️⃣ Dataset Preprocessing (Common)

### 🔹 Normalize images

```python
x_train = x_train / 255.0
x_test  = x_test / 255.0
```

---

### 🔹 One-hot encode labels

```python
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)
```

---

## 7️⃣ Train-Test Split (Custom data)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 8️⃣ Common Mistakes 🚨

❌ Dataset normalize না করা
❌ Wrong input shape
❌ Label encoding mismatch
❌ RGB vs grayscale confusion

---

## 9️⃣ Quick Cheat Sheet 🧠

| Data type | Loader                         |
| --------- | ------------------------------ |
| Built-in  | `tf.keras.datasets`            |
| CSV       | `pandas.read_csv`              |
| Images    | `image_dataset_from_directory` |
| Text      | `Tokenizer`                    |
| Big data  | `tf.data.Dataset`              |

---

## 🔟 End-to-End Example (MNIST)

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test / 255.0
```

---

## 🔑 One-line Summary

> **Dataset loading হলো ML pipeline-এর foundation—এটা ঠিক হলে বাকি সব সহজ হয়**

---

