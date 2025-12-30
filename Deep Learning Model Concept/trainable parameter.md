---

## 1️⃣ Trainable Parameter কী?

Neural Network-এ যেগুলো **training এর সময় update হয়**, সেগুলোকে বলে **trainable parameters**:

* **Weights (W)**
* **Biases (b)**

👉 এগুলোর মান backpropagation দিয়ে পরিবর্তন হয়।

---

## 2️⃣ `show_trainable = true` এর মানে

যখন

```
show_trainable = true
```

দেওয়া হয়, তখন:

✅ **শুধু trainable parameter গুলো দেখাবে**
❌ non-trainable (freeze করা) parameter দেখাবে না

---

## 3️⃣ TensorFlow উদাহরণ

### Model বানানো

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])
```

### Model Summary (Trainable parameter দেখাবে)

```python
model.summary()
```

📌 Output-এ যা দেখাবে:

* Layer name
* Output shape
* **Param # (trainable)**

---

## 4️⃣ Trainable Parameter আলাদা করে দেখা

### মোট trainable parameter

```python
model.count_params()
```

---

### শুধু trainable weights দেখা

```python
for var in model.trainable_variables:
    print(var.name, var.shape)
```

📌 এখানে দেখাবে:

* `kernel` (weights)
* `bias`

---

## 5️⃣ Trainable vs Non-Trainable Example

```python
layer = tf.keras.layers.Dense(10)
layer.trainable = False
```

এখন যদি:

```python
show_trainable = true
```

দেওয়া হয়,

👉 এই layer-এর parameter দেখাবে না
কারণ এগুলো **train হচ্ছে না**

---

## 6️⃣ সহজ ভাষায় এক লাইনে

> **`show_trainable = true` মানে হলো — training এর সময় যেসব parameter (weights ও bias) update হয়েছে, সেগুলো দেখানো হবে।**

---

## 7️⃣ Exam-Ready Short Answer ✍️

> *When `show_trainable = true`, only the trainable parameters (weights and biases that are updated during training) of the neural network are displayed.*

---

