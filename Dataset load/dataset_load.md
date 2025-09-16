
# MNIST `load_data()` – Tuple Unpacking Guide

## কী রিটার্ন করে?

`tensorflow.keras.datasets.mnist.load_data()` এই আকারে ডেটা ফেরত দেয়:

```python
((x_train, y_train), (x_test, y_test))
```

অর্থাৎ—**দুটি টাপল** (train ও test), এবং প্রতিটা টাপলে আছে **দুটি** করে আইটেম: ইমেজ ও লেবেল।

* `x_train.shape == (60000, 28, 28)`
* `y_train.shape == (60000,)`
* `x_test.shape  == (10000, 28, 28)`
* `y_test.shape  == (10000,)`

## কেন দুই স্তরের ব্র্যাকেট?

কারণ ডান পাশে রিটার্ন ভ্যালুটি **নেস্টেড টাপল**। তাই বামে **নেস্টেড আনপ্যাকিং** করতে হয়:

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

এতে প্রথম টাপলটি `x_train, y_train`-এ, আর দ্বিতীয় টাপলটি `x_test, y_test`-এ খুলে যায়।

## যদি এক লাইনে চারটা ভ্যারিয়েবল দাও?

এভাবে লিখলে—

```python
(x_train, y_train, x_test, y_test) = mnist.load_data()
```

এরর হবে, কারণ ডান পাশে আছে **২টি** আইটেম (দুটি টাপল), কিন্তু বাঁ পাশে চাইছ **৪টি** ভ্যারিয়েবল। পাইথন বলবে:

```
ValueError: not enough values to unpack (expected 4, got 2)
```

## বিকল্প, ধাপে ধাপে আনপ্যাক

### Option A: দুই ধাপে

```python
(train_tuple, test_tuple) = mnist.load_data()
x_train, y_train = train_tuple
x_test,  y_test  = test_tuple
```

### Option B: ইনডেক্সিং

```python
data = mnist.load_data()
x_train, y_train = data[0]  # train tuple
x_test,  y_test  = data[1]  # test tuple
```

## দ্রুত স্যানিটি-চেক

লোড করার পর শেপ/টাইপ মিলিয়ে দেখো:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape,  y_test.shape)   # (10000, 28, 28) (10000,)
print(x_train.dtype)                 # uint8
```

## সাধারণ ভুল ও সমাধান

* ❌ `x_train, y_train, x_test, y_test = mnist.load_data()`

  * ✅ ঠিক: `(x_train, y_train), (x_test, y_test) = mnist.load_data()`
* ❌ ব্র্যাকেট কম/বেশি দেওয়া, ফলে `ValueError`.

  * ✅ রিটার্নের স্ট্রাকচার মাথায় রেখে **নেস্টেড ব্র্যাকেট** বসাও।
* ❌ `y_train`/`y_test` one-hot না করে সরাসরি softmax আউটপুটে ট্রেন।

  * ✅ `to_categorical(y, 10)` করে one-hot বানাও (ট্রেনিং ধাপে দরকার হলে)।

## ছোট উদাহরণ (কেন নেস্টেড লাগে—পিওর পাইথন)

```python
def f():
    return (('XA', 'YA'), ('XB', 'YB'))  # নেস্টেড tuple

# সঠিক:
(a1, b1), (a2, b2) = f()  # OK

# ভুল:
# a1, b1, a2, b2 = f()   # ValueError: expected 4, got 2
```

**সারসংক্ষেপ:** `mnist.load_data()` নেস্টেড টাপল ফেরত দেয়, তাই **দুটি ব্র্যাকেট** দিয়ে নেস্টেড আনপ্যাকিং করাই সঠিক পদ্ধতি।



---

# 1) TensorFlow / Keras built-in datasets (numpy arrays)

**রিটার্ন ফরম্যাট:** `((x_train, y_train), (x_test, y_test))` — তাই nested unpacking লাগে।

### MNIST / Fashion-MNIST

```python
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Fashion-MNIST
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess (common)
x_train = (x_train.astype("float32")/255.0)[..., None]  # (N,28,28,1)
x_test  = (x_test.astype("float32")/255.0)[..., None]
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)
```

### CIFAR-10 / CIFAR-100 (RGB 32×32)

```python
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

# CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# CIFAR-100
# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

x_train = x_train.astype("float32")/255.0  # (N,32,32,3)
x_test  = x_test.astype("float32")/255.0
y_train_cat = to_categorical(y_train, 10)  # CIFAR-10 → 10
y_test_cat  = to_categorical(y_test, 10)
```

### টেক্সট: IMDB sentiment (sequence of word indices)

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(num_words, maxlen) = (20000, 200)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test  = pad_sequences(x_test,  maxlen=maxlen)
# y গুলো 0/1 (binary labels)
```

---

# 2) TensorFlow Datasets (TFDS) — `tf.data.Dataset`

**রিটার্ন ফরম্যাট (সুপারভাইজড):** `tf.data.Dataset` যার প্রতিটা উদাহরণ `(features, label)`।

```python
import tensorflow_datasets as tfds
import tensorflow as tf

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    as_supervised=True,  # (image, label)
    with_info=True
)

# Normalize + batch
def norm_map(img, lbl):
    img = tf.cast(img, tf.float32)/255.0
    img = tf.expand_dims(img, -1)   # (28,28,1)
    return img, lbl

BATCH = 128
ds_train = ds_train.map(norm_map).shuffle(10_000).batch(BATCH).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(norm_map).batch(BATCH).prefetch(tf.data.AUTOTUNE)
```

> **নোট:** `ds_info.features` থেকে ক্লাসনেম, শেপ, কাউন্ট জানা যায়।

---

# 3) PyTorch `torchvision` datasets (+ DataLoader)

**রিটার্ন ফরম্যাট:** `Dataset` অবজেক্ট; `DataLoader` দিয়ে batch ইটারেট করবে: `(images, labels)` (টেন্সর)।

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                  # [0,1], shape: (C,H,W)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std (grayscale)
])

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2)

# Iterate
for images, labels in train_loader:
    # images: [B,1,28,28], labels: [B]
    break
```

CIFAR-10:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))  # CIFAR-10 stats
])
train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
```

---

# 4) scikit-learn datasets (tabular/classic)

**রিটার্ন ফরম্যাট:** `Bunch` (dict-like), সাধারণত `.data` (X), `.target` (y)।

### Iris / Digits (লোকাল)

```python
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target   # X: (150,4), y: (150,)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### OpenML থেকে MNIST (বড়; নেট লাগবে)

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype('int64')  # X: (70000,784)

# reshape করলে (N,28,28) পাবে:
# X_img = X.reshape(-1, 28, 28)
```

---

# 5) Hugging Face `datasets` (ইমেজ/টেক্সট/মাল্টিমোডাল)

**রিটার্ন ফরম্যাট:** `DatasetDict` → split-wise `Dataset`; কলাম-নেমসহ টেবল-সদৃশ।

```python
from datasets import load_dataset

ds = load_dataset("mnist")  # splits: 'train', 'test'
print(ds)
# ds['train'][0] -> {'image': PIL.Image.Image, 'label': int}

# টেন্সর ফরম্যাট (PyTorch):
ds.set_format(type="torch", columns=["image","label"])
batch = ds["train"][:32]
```

CIFAR-10:

```python
ds = load_dataset("cifar10")
```

---

# 6) নিজের ইমেজ ফোল্ডার (Keras): `image_dataset_from_directory`

**ডিরেক্টরি স্ট্রাকচার**:

```
data/
  train/
    classA/ img1.jpg ...
    classB/ ...
  val/
    classA/ ...
    classB/ ...
```

```python
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train", image_size=(224,224), batch_size=32,
    label_mode="int", shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val", image_size=(224,224), batch_size=32,
    label_mode="int", shuffle=False
)

# Normalize pipeline
normalizer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x,y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)
```

> **টিপ:** এক ফোল্ডার থেকে train/val split দরকার হলে `validation_split` + `subset` সহ একই API ব্যবহার করতে পারো।

---

# 7) CSV/ট্যাবুলার ডেটা (pandas → NumPy/TF/PyTorch)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
X = df.drop(columns=["label"]).values
y = df["label"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)
```

---

## আনপ্যাকিং ও রিটার্ন-টাইপ—ফাস্ট রিমাইন্ডার

* **Keras built-in (numpy arrays):** `((x_train,y_train),(x_test,y_test))`

  * ✅ `(x_train, y_train), (x_test, y_test) = mnist.load_data()`
* **TFDS / Keras directory datasets:** `tf.data.Dataset` (ইটারেবল ব্যাচ)
* **torchvision:** `Dataset` → `DataLoader` থেকে `(images, labels)`
* **sklearn:** `Bunch` → `.data`/`.target`
* **Hugging Face:** `DatasetDict`/`Dataset` টেবল-সদৃশ; কলাম-বেইজড অ্যাক্সেস

---

## সাধারণ প্রিপ্রসেস টিপস

* **ইমেজ:** `/255.0` নরমালাইজ, শেপ ঠিক করো (Keras: `(H,W,1/3)`, PyTorch: `(C,H,W)`), প্রয়োজন হলে mean/std normalize।
* **লেবেল:** Keras-এ softmax ট্রেনিং হলে `to_categorical` (one-hot) দরকার। PyTorch-এ `CrossEntropyLoss` নিলে integer class labels রাখাই ঠিক।
* **শাফল/ব্যাচিং:** TFDS/`image_dataset_from_directory`/`DataLoader`—সব ক্ষেত্রেই শাফল+ব্যাচ সেট করো।
* **ভ্যালিডেশন স্প্লিট:** Keras এ `validation_split` (in-memory arrays), বা আলাদা ভ্যালিডেশন ফোল্ডার/স্প্লিট।


