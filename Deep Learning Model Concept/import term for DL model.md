

---

# 🧠 Deep Learning – Important Terms (Overview)

প্রথমে **সব term-এর লিস্ট** 👇

1️⃣ Dataset

2️⃣ Sample

3️⃣ Feature

4️⃣ Label / Target

5️⃣ Model

6️⃣ Parameters

7️⃣ Hyperparameters

8️⃣ Epoch

9️⃣ Batch

🔟 Batch Size

1️⃣1️⃣ Iteration / Step

1️⃣2️⃣ Loss / Cost

1️⃣3️⃣ Optimizer

1️⃣4️⃣ Learning Rate

1️⃣5️⃣ Forward Propagation

1️⃣6️⃣ Backpropagation

1️⃣7️⃣ Gradient

1️⃣8️⃣ Activation Function

1️⃣9️⃣ Overfitting

2️⃣0️⃣ Underfitting

2️⃣1️⃣ Train / Validation / Test Set

2️⃣2️⃣ Metrics

2️⃣3️⃣ Callback (EarlyStopping ইত্যাদি)

এখন একে একে ব্যাখ্যা করছি 👇

---

# 1️⃣ Dataset

### 🔹 Definition

👉 Training-এর জন্য ব্যবহার করা **পুরো ডেটার সংগ্রহ**।

### 🔹 Example

```text
x = [1, 2, 3, 4]
y = [3, 5, 7, 9]
```

---

# 2️⃣ Sample

### 🔹 Definition

👉 Dataset-এর **একটা একক ডেটা পয়েন্ট**।

### 🔹 Example

```text
Sample: x = 3 , y = 7
```

---

# 3️⃣ Feature

### 🔹 Definition

👉 Input-এর **individual attribute / variable**।

### 🔹 Example

```python
x = [age, height, weight]
```

---

# 4️⃣ Label / Target

### 🔹 Definition

👉 Model যেটা predict করতে শেখে।

### 🔹 Example

```python
y = house_price
```

---

# 5️⃣ Model

### 🔹 Definition

👉 Mathematical function যেটা **input → output mapping** শেখে।

### 🔹 Example

```text
y = wx + b
```

---

# 6️⃣ Parameters

### 🔹 Definition

👉 Model-এর ভেতরের **learnable values** (training-এ update হয়)।

### 🔹 Example

```text
w (weight), b (bias)
```

---

# 7️⃣ Hyperparameters

### 🔹 Definition

👉 Training শুরু করার আগেই ঠিক করা মান,
যেগুলো model নিজে শেখে না।

### 🔹 Example

```text
learning rate, batch size, epochs
```

---

# 8️⃣ Epoch

### 🔹 Definition

👉 **পুরো training dataset একবার complete করে দেখা**।

### 🔹 Example

```python
epochs = 50
```

📌 মানে model পুরো dataset 50 বার দেখবে।

---

# 9️⃣ Batch

### 🔹 Definition

👉 Dataset-এর **ছোট ছোট অংশ**, যেগুলো দিয়ে training হয়।

### 🔹 Example

```text
Batch = 32 samples
```

---

# 🔟 Batch Size

### 🔹 Definition

👉 একবারে model কতগুলো sample নিয়ে কাজ করবে।

### 🔹 Example

```python
batch_size = 32
```

📌 Smaller batch → stable but slow
📌 Larger batch → fast but noisy gradient

---

# 1️⃣1️⃣ Iteration / Step

### 🔹 Definition

👉 **একটা batch process হওয়া = 1 iteration**।

### 🔹 Formula

```text
iterations per epoch = total_samples / batch_size
```

---

# 1️⃣2️⃣ Loss / Cost Function

### 🔹 Definition

👉 Model কতটা ভুল করছে তা **সংখ্যায় প্রকাশ করে**।

### 🔹 Example

```python
loss = Mean Squared Error
```

---

# 1️⃣3️⃣ Optimizer

### 🔹 Definition

👉 Loss কমানোর জন্য **weights update করার নিয়ম**।

### 🔹 Example

```python
optimizer = Adam()
```

---

# 1️⃣4️⃣ Learning Rate

### 🔹 Definition

👉 Weight কত বড় step-এ update হবে।

### 🔹 Example

```python
learning_rate = 0.001
```

📌 খুব বড় → model ভেঙে যাবে
📌 খুব ছোট → training slow

---

# 1️⃣5️⃣ Forward Propagation

### 🔹 Definition

👉 Input থেকে output calculate করার প্রক্রিয়া।

### 🔹 Flow

```text
Input → Layers → Output
```

---

# 1️⃣6️⃣ Backpropagation

### 🔹 Definition

👉 Output error থেকে **weights update করার প্রক্রিয়া**।

### 🔹 Flow

```text
Loss → Gradient → Weight update
```

---

# 1️⃣7️⃣ Gradient

### 🔹 Definition

👉 Loss কত দ্রুত বাড়ছে/কমছে তার দিক ও পরিমাণ।

📌 Optimizer gradient ব্যবহার করে।

---

# 1️⃣8️⃣ Activation Function

### 🔹 Definition

👉 Model-কে **non-linear** বানায়।

### 🔹 Example

```python
ReLU, Sigmoid, Softmax
```

---

# 1️⃣9️⃣ Overfitting

### 🔹 Definition

👉 Model training data খুব ভালো শেখে
কিন্তু নতুন data-তে খারাপ করে।

📌 Train accuracy ↑, Test accuracy ↓

---

# 2️⃣0️⃣ Underfitting

### 🔹 Definition

👉 Model খুব simple, কিছুই ভালো শেখে না।

📌 Train ↓, Test ↓

---

# 2️⃣1️⃣ Train / Validation / Test Set

### 🔹 Definition

| Set        | কাজ           |
| ---------- | ------------- |
| Train      | শেখা          |
| Validation | tune করা      |
| Test       | final পরীক্ষা |

---

# 2️⃣2️⃣ Metrics

### 🔹 Definition

👉 Performance measure (training প্রভাব ফেলে না)।

### 🔹 Example

```python
accuracy, MAE
```

---

# 2️⃣3️⃣ Callback

### 🔹 Definition

👉 Training চলাকালীন **অতিরিক্ত control** দেয়।

### 🔹 Example

```python
EarlyStopping
```

---

# 🧠 One-Page Memory Trick

```text
Epoch → Dataset কতবার দেখবে
Batch → একসাথে কত data
Iteration → এক batch process
Loss → কত ভুল
Optimizer → ভুল ঠিক করা
```

---

# ✅ Interview One-liners (Very Important)

* **Epoch** = one full pass over data
* **Batch size** = samples per update
* **Optimizer** = weight update rule
* **Loss** = error measure
* **Overfitting** = memorization

---

