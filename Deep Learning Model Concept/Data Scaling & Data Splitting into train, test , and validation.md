
---

# 🧾 Data Scaling & Data Splitting — Full Documentation

তোমার কোড (reference):

```python
# scale input and output  
max_x = x.max()
max_y = y.max()

x_scaled = x / max_x 
y_scaled = y / max_y

x_scaled = x_scaled.reshape(-1,1)
y_scaled = y_scaled.reshape(-1,1)

# ----- Split data (70% train ,10% val , 20% test )

x_train_val, x_test, y_train_val, y_test = train_test_split(
    x_scaled, y_scaled, test_size=0.2, random_state=42 
) 

x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.125, random_state=42
)
```

---

# 🔷 PART 1: Data Scaling (Feature Scaling)

## 🔹 Data Scaling কী?

👉 **Data scaling** মানে হলো:

> ডেটাকে একটা ছোট, নির্দিষ্ট range-এর মধ্যে আনা
> যাতে model সহজে শিখতে পারে

---

## 🔹 কেন scaling দরকার?

### Neural Network / ML model-এ সমস্যা হয় যদি scaling না করা হয়:

* Gradient slow হয়
* Training unstable হয়
* Loss বেশি fluctuation করে
* Large value dominate করে

📌 তাই **almost সব ML/DL model-এ scaling দরকার**

---

## 🔹 তোমার Scaling Method: Max Scaling

```python
x_scaled = x / max_x
y_scaled = y / max_y
```

### এটাকে বলে:

> **Max Scaling** বা **Normalization (0–1)**

---

## 🔹 কী হচ্ছে এখানে?

ধরা যাক:

```python
x = [0, 50, 100]
max_x = 100
```

তাহলে:

```python
x_scaled = [0/100, 50/100, 100/100]
         = [0.0, 0.5, 1.0]
```

✔ সব value এখন 0–1 এর মধ্যে
✔ Training stable

---

## 🔹 কেন input এবং output দুটোই scale করা হয়েছে?

```python
x_scaled = x / max_x
y_scaled = y / max_y
```

✔ Input scale → model fast শেখে
✔ Output scale → loss stable থাকে

📌 Regression problem-এ output scaling খুব গুরুত্বপূর্ণ

---

## 🔹 reshape কেন করা হয়েছে?

```python
x_scaled = x_scaled.reshape(-1,1)
```

### কারণ:

Keras / scikit-learn expect করে:

```
(samples, features)
```

* `-1` → NumPy নিজে sample সংখ্যা ঠিক করে
* `1` → 1 feature

❌ reshape না করলে:

```
ValueError / shape mismatch
```

---

## 🔹 Scaling না করলে কী হতো?

| Without scaling   | With scaling       |
| ----------------- | ------------------ |
| Training slow     | Training fast      |
| Gradient unstable | Stable gradient    |
| Poor convergence  | Smooth convergence |

---

# 🔷 PART 2: Data Splitting (Train / Validation / Test)

---

## 🔹 Data Split কেন দরকার?

👉 Model-কে ৩টা আলাদা জিনিস শেখাতে হয়:

1️⃣ Train → শেখা
2️⃣ Validation → tune করা
3️⃣ Test → final পরীক্ষা

📌 **Test data কখনো training-এ ব্যবহার করা যাবে না**

---

# 🔹 `train_test_split()` — Full Documentation

Import:

```python
from sklearn.model_selection import train_test_split
```

---

## 🔹 Full Syntax

```python
train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None
)
```

---

# 🔹 Mandatory Parameters (অবশ্যই লাগবে)

### ✅ `*arrays`

```python
train_test_split(x, y)
```

* Split করতে চাও এমন array গুলো
* একসাথে pass করলে alignment ঠিক থাকে

❌ একটাই দিলে target mismatch হতে পারে

---

### ✅ `test_size` অথবা `train_size` (একটা দিলেই চলবে)

```python
test_size=0.2
```

| মান   | অর্থ            |
| ----- | --------------- |
| `0.2` | 20% test        |
| `0.3` | 30% test        |
| `100` | 100 sample test |

❌ দুটোই না দিলে → error

---

# 🔹 Optional Parameters (কিন্তু খুব গুরুত্বপূর্ণ)

### `random_state`

```python
random_state=42
```

👉 Seed value

* Same split বারবার পেতে
* Reproducibility

না দিলে → প্রতিবার আলাদা split

---

### `shuffle`

```python
shuffle=True
```

* Data shuffle হবে কিনা
* Default: `True`

❌ Time-series data হলে `False`

---

### `stratify`

```python
stratify=y
```

* Classification এ class ratio বজায় রাখে
* Regression এ সাধারণত লাগে না

---

# 🔷 তোমার Split Logic (Detailed Breakdown)

---

## Step 1️⃣: Train+Val vs Test (80% / 20%)

```python
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x_scaled, y_scaled, test_size=0.2, random_state=42
)
```

✔ 20% → test
✔ 80% → train+val

---

## Step 2️⃣: Train vs Validation (70% / 10%)

```python
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.125, random_state=42
)
```

### কেন `0.125`?

কারণ:

```
80% × 0.125 = 10%
```

✔ Final split:

* Train = 70%
* Validation = 10%
* Test = 20%

---

## 🔹 Print output explanation

```python
print(len(x_train))  # ~70%
print(len(x_val))    # ~10%
print(len(x_test))   # ~20%
```

📌 Correct ML pipeline

---

# 🔥 Common Mistakes (VERY IMPORTANT)

❌ Scaling করার আগে split না করা
❌ Test data থেকে `max_x` বের করা (data leakage)
❌ `random_state` না দেওয়া
❌ Validation data দিয়ে model train করা

---

# 🧠 Best Practice (Industry Standard)

```text
1. Split data
2. Fit scaler on train only
3. Transform val & test
```

(তোমার উদাহরণ simple demo, তাই acceptable)

---

# 🧪 Alternative Scaling Methods (Reference)

| Method          | Use           |
| --------------- | ------------- |
| Min-Max Scaling | 0–1 range     |
| StandardScaler  | mean=0, std=1 |
| RobustScaler    | outlier safe  |

---

# 📌 Summary Table (Exam Ready)

| Topic          | Explanation      |
| -------------- | ---------------- |
| Scaling        | Normalize values |
| Why scale      | Stable training  |
| `test_size`    | Test ratio       |
| `random_state` | Reproducibility  |
| `shuffle`      | Data mixing      |
| `stratify`     | Class balance    |

---

## 🧠 One-line Interview Answer

> Data scaling normalizes feature ranges for stable training, and `train_test_split` separates data into train, validation, and test sets to fairly evaluate model performance.

---


---

# ✅ FULL WORKING CODE

## (Data Scaling + Train/Val/Test Split with Explanation)

```python
# =========================
# Imports
# =========================
import numpy as np
from sklearn.model_selection import train_test_split


# =========================
# Example polynomial function
# =========================
def my_polynomial(x):
    # y = 3x^2 + 2x + 1 (example function)
    return 3 * x**2 + 2 * x + 1


# =========================
# Data Processing Function
# =========================
def data_process(n=1000, random_seed=42):
    """
    n           : total number of samples
    random_seed : reproducibility control
    """

    # ---------- Random seed (reproducibility)
    np.random.seed(random_seed)

    # ---------- Generate random x values
    x = np.random.randint(0, n, n).astype(np.float32)

    # ---------- Generate y using polynomial function
    y = my_polynomial(x).astype(np.float32)

    # =========================
    # DATA SCALING
    # =========================

    # Maximum values
    max_x = x.max()
    max_y = y.max()

    # Scale to 0–1 range
    x_scaled = x / max_x
    y_scaled = y / max_y

    # Reshape for ML models (samples, features)
    x_scaled = x_scaled.reshape(-1, 1)
    y_scaled = y_scaled.reshape(-1, 1)

    # =========================
    # DATA SPLITTING
    # =========================

    # ---- Step 1: Split into (train+val) and test
    # 80% train+val, 20% test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_scaled,
        y_scaled,
        test_size=0.2,          # 20% test data
        random_state=random_seed
    )

    # ---- Step 2: Split train+val into train and validation
    # 80% of remaining → 70% train, 10% val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.125,        # 10% of total data
        random_state=random_seed
    )

    # =========================
    # Print dataset sizes
    # =========================
    print(f"Train samples      : {len(x_train)}")
    print(f"Validation samples : {len(x_val)}")
    print(f"Test samples       : {len(x_test)}")

    return x_train, y_train, x_val, y_val, x_test, y_test


# =========================
# Run the function
# =========================
x_train, y_train, x_val, y_val, x_test, y_test = data_process()
```

---

# 🧠 STEP-BY-STEP EXPLANATION

---

## 🔹 1. Random Seed

```python
np.random.seed(random_seed)
```

👉 একই seed দিলে **same random data বারবার পাওয়া যায়**

✔ debugging
✔ experiment reproducibility

---

## 🔹 2. Data Generation

```python
x = np.random.randint(0, n, n)
y = my_polynomial(x)
```

* `x` → random input values
* `y` → known mathematical relation

📌 supervised learning-এর perfect example

---

## 🔹 3. Data Scaling (WHY?)

```python
x_scaled = x / max_x
y_scaled = y / max_y
```

### কেন দরকার?

* Neural network বড় সংখ্যা ভালো handle করে না
* Gradient stable থাকে
* Faster convergence

📌 এটাকে বলে **Max Scaling (0–1 normalization)**

---

## 🔹 4. Reshape (VERY IMPORTANT)

```python
x_scaled.reshape(-1, 1)
```

👉 ML model চায়:

```
(samples, features)
```

❌ reshape না করলে shape error আসবে

---

## 🔹 5. `train_test_split()` — First Split

```python
test_size=0.2
```

👉 20% data → **Test set**

| Parameter            | কাজ            |
| -------------------- | -------------- |
| `x_scaled, y_scaled` | input + target |
| `test_size=0.2`      | 20% test       |
| `random_state`       | same split     |

---

## 🔹 6. Second Split (Train vs Validation)

```python
test_size=0.125
```

👉 কারণ:

```
0.125 × 80% ≈ 10%
```

✔ Final ratio:

* Train = 70%
* Validation = 10%
* Test = 20%

---

## 🔹 7. Why Validation Data?

* Model tune করার জন্য
* Overfitting ধরার জন্য
* Test data untouched রাখতে

---

# 📊 FINAL DATA DISTRIBUTION

| Dataset    | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 10%        |
| Test       | 20%        |

---

# ⚠️ COMMON MISTAKES (Interview Point)

❌ Scaling test data using test statistics
❌ Validation data দিয়ে training করা
❌ Test data repeatedly check করা
❌ `random_state` না দেওয়া

---

# 🧠 ONE-LINE INTERVIEW ANSWER

> We scale data to stabilize learning and split it into train, validation, and test sets to train, tune, and fairly evaluate a machine learning model.

---

