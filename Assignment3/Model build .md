
---

# 🧾 TensorFlow Keras Model Making

## (Functional API) — Full Detailed Documentation

---

# 🔷 Model বানাতে মোটামুটি ৪টা জিনিস লাগে

1️⃣ `Input()`
2️⃣ `Dense()` (বা অন্য layers)
3️⃣ `outputs` tensor
4️⃣ `Model()`

তারপর:

* `compile()`
* `fit()`
* `predict()`

---

# 1️⃣ `Input()` — Input Layer (সবচেয়ে প্রথম দরজা)

## 🔹 Input() কী?

👉 `Input()` বলে দেয়:

> “আমার model কেমন shape-এর data নেবে”

⚠️ **Input data নেয় না**, শুধু **shape define করে**

---

## 🔹 Input() Full Syntax

```python
Input(
    shape,              # REQUIRED
    batch_size=None,    # optional
    name=None,          # optional
    dtype=None,         # optional
    sparse=False,       # optional
    ragged=False        # optional
)
```

---

## 🔹 Mandatory Parameter

### ✅ `shape` (অবশ্যই লাগবে)

```python
Input(shape=(1,))
```

| মান           | অর্থ         |
| ------------- | ------------ |
| `(1,)`        | ১টা feature  |
| `(10,)`       | ১০টা feature |
| `(28, 28, 1)` | image input  |

❌ shape না দিলে:

```python
Input()
```

➡️ **Error আসবে**

---

## 🔹 Optional Parameters (না নিলে কী হবে?)

### `batch_size`

```python
Input(shape=(1,), batch_size=32)
```

* Fixed batch size enforce করে
* না নিলে → TensorFlow নিজে manage করে (BEST PRACTICE)

👉 **সাধারণত নেয়া লাগে না**

---

### `name`

```python
Input(shape=(1,), name="input_layer")
```

* Graph / summary readable হয়
* Debugging সহজ

না নিলে → auto নাম (`input_1`)

---

### `dtype`

```python
Input(shape=(1,), dtype="float32")
```

* Data type specify
* না নিলে → `float32` default

---

## 🔹 Minimum correct Input

```python
inputs = Input(shape=(1,))
```

📌 **এটাই সবচেয়ে common**

---

# 2️⃣ `Dense()` — Fully Connected Layer

---

## 🔹 Dense Layer কী করে?

👉 Dense layer শেখে:

```
output = activation(Wx + b)
```

* `W` = weights
* `b` = bias
* `activation` = non-linearity

---

## 🔹 Dense() Full Syntax (Complete)

```python
Dense(
    units,                       # REQUIRED
    activation=None,             # optional
    use_bias=True,               # optional
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    name=None
)
```

---

## 🔹 Mandatory Parameter

### ✅ `units` (অবশ্যই লাগবে)

```python
Dense(8)
```

| মান | অর্থ          |
| --- | ------------- |
| 8   | 8টা neuron    |
| 1   | single output |
| 64  | 64 neuron     |

❌ units না দিলে → error

---

## 🔹 Optional Parameters (খুব গুরুত্বপূর্ণ)

### `activation`

```python
Dense(16, activation='relu')
```

| Activation | না নিলে কী হবে    |
| ---------- | ----------------- |
| `relu`     | non-linearity যোগ |
| না নিলে    | linear activation |

📌 Hidden layer এ activation না দিলে → model useless হয়ে যায়

---

### `use_bias`

```python
Dense(8, use_bias=False)
```

* Bias term বাদ দেয়
* না নিলে → bias থাকবে (default)

---

### `name`

```python
Dense(8, name="hidden_layer1")
```

* Layer identify করা সহজ
* Summary সুন্দর হয়

না নিলে → auto name (`dense_1`)

---

## 🔹 Minimum correct Dense

```python
Dense(8)
```

📌 কিন্তু hidden layer হলে **activation নেওয়া উচিত**

---

## 🔹 Hidden Layer vs Output Layer

### Hidden layer

```python
Dense(16, activation='relu')
```

### Output layer (Regression)

```python
Dense(1, activation='linear')
```

### Output layer (Binary classification)

```python
Dense(1, activation='sigmoid')
```

---

# 3️⃣ `outputs` — Final Tensor

```python
outputs = Dense(1)(x)
```

👉 এটা **Tensor**, model না

* Shape: `(batch_size, 1)`
* Model কী return করবে সেটা define করে

📌 এখানে ভুল হলে পুরো model ভুল

---

# 4️⃣ `Model()` — Model Container (সবচেয়ে গুরুত্বপূর্ণ)

---

## 🔹 Model() Full Syntax

```python
Model(
    inputs,              # REQUIRED
    outputs,             # REQUIRED
    name=None,            # optional
    trainable=True        # optional
)
```

---

## 🔹 Mandatory Parameters

### ✅ `inputs`

```python
inputs = Input(shape=(1,))
```

* Model কোথা থেকে শুরু করবে

---

### ✅ `outputs`

```python
outputs = Dense(1)(x)
```

* Model কোথায় শেষ হবে

❌ inputs বা outputs না দিলে → model বানবে না

---

## 🔹 Optional Parameters

### `name`

```python
Model(inputs, outputs, name="linear_regression_model")
```

* Model identify করা সহজ

---

### `trainable`

```python
Model(inputs, outputs, trainable=False)
```

* Weight freeze করে
* Transfer learning এ লাগে

---

## 🔹 Minimum correct Model

```python
model = Model(inputs, outputs)
```

---

# 5️⃣ Full Minimal Example (সব mandatory)

```python
inputs = Input(shape=(1,))
x = Dense(8, activation='relu')(inputs)
outputs = Dense(1)(x)

model = Model(inputs, outputs)
```

---

# 6️⃣ Full Recommended Example (Best Practice)

```python
inputs = Input(shape=(1,), name="input_layer")

x = Dense(8, activation='relu', name="hidden1")(inputs)
x = Dense(16, activation='relu', name="hidden2")(x)
x = Dense(4, activation='relu', name="hidden3")(x)

outputs = Dense(1, activation='linear', name="output")(x)

model = Model(inputs=inputs, outputs=outputs, name="regression_model")
```

---

# 7️⃣ Model Compile (অবশ্যই লাগবে training এর আগে)

```python
model.compile(
    optimizer='adam',
    loss='mse'
)
```

| Parameter | Mandatory? |
| --------- | ---------- |
| optimizer | ✅          |
| loss      | ✅          |
| metrics   | ❌          |

---

# 8️⃣ Summary Table (Exam Ready)

| Component        | Mandatory | না নিলে কী হবে    |
| ---------------- | --------- | ----------------- |
| Input.shape      | ✅         | Error             |
| Dense.units      | ✅         | Error             |
| Dense.activation | ❌         | Linear behaviour  |
| Dense.name       | ❌         | Auto name         |
| Model.inputs     | ✅         | Error             |
| Model.outputs    | ✅         | Error             |
| compile()        | ✅         | train করা যাবে না |

---

# 🧠 Golden Rule (সবচেয়ে গুরুত্বপূর্ণ)

> **Functional API তে layer হলো function, আর tensor হলো data flow।**

---

## 🎯 One-line Interview Answer

> A Keras model is defined by explicitly connecting input tensors to output tensors using layers as callable functions.

---



---

# 🔒 Weight Freeze মানে কী?

👉 **Weight freeze** মানে হলো—

> **model-এর কিছু layer-এর weight training সময় আর update হবে না**

অর্থাৎ:

* Backpropagation হবে ❌
* Gradient apply হবে ❌
* Weight আগের মতোই থাকবে ✅

📌 Model ওই layer গুলোকে **শুধু ব্যবহার করবে, শিখবে না**।

---

## 🧠 Simple analogy (Real-life)

ধরো তুমি:

* আগে থেকেই **English grammar** জানো
* এখন **IELTS speaking** শিখছো

👉 Grammar তুমি আবার শেখো না
👉 Grammar শুধু **ব্যবহার করো**

✔ Grammar = **frozen weights**
✔ Speaking practice = **trainable layers**

---

# 🔹 Neural Network Context-এ Weight Freeze

একটা neural network এ সাধারণত থাকে:

1️⃣ Early layers → basic feature শিখে
2️⃣ Middle layers → complex pattern
3️⃣ Last layers → task-specific decision

Transfer Learning এ:

* Early + middle layers → **freeze**
* Last layers → **train**

---

# 🔹 `trainable=False` মানে কী?

```python
Model(inputs, outputs, trainable=False)
```

👉 এর মানে:

* Model-এর **সব layer** frozen
* কোনো weight update হবে না

⚠️ এটা সাধারণত **পুরো model freeze** করতে ব্যবহৃত হয়

---

## 🔹 Layer-wise freeze (সবচেয়ে common)

```python
for layer in model.layers:
    layer.trainable = False
```

✔ Pretrained feature extractor freeze
✔ New head train করা যায়

---

# 🔹 Transfer Learning Workflow (Step-by-Step)

### Step 1️⃣ Pretrained model নাও

```python
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False
)
```

---

### Step 2️⃣ Weight freeze করো

```python
base_model.trainable = False
```

📌 এখন VGG16 feature extractor হিসেবে কাজ করবে

---

### Step 3️⃣ New layers যোগ করো

```python
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(base_model.input, outputs)
```

---

### Step 4️⃣ Compile & Train

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

👉 Training এ:

* Pretrained weights unchanged
* New layers train হবে

---

# 🔍 Freeze না করলে কী হতো?

```python
base_model.trainable = True
```

❌ সমস্যা:

* Pretrained knowledge নষ্ট হতে পারে
* Small dataset এ overfitting
* Training slow

---

# 🔹 Partial Freeze (Advanced)

```python
for layer in base_model.layers[:10]:
    layer.trainable = False

for layer in base_model.layers[10:]:
    layer.trainable = True
```

👉 Early layers freeze
👉 Deeper layers fine-tune

📌 একে বলে **Fine-tuning**

---

# 🔥 Why weight freeze is important?

| Benefit          | Explanation           |
| ---------------- | --------------------- |
| Faster training  | কম parameter update   |
| Less overfitting | Small data safe       |
| Reuse knowledge  | Pretrained features   |
| Stable learning  | Gradient explosion কম |

---

# 🧠 Backpropagation Perspective

### Normal training:

```
Loss → Gradient → Update all weights
```

### With freeze:

```
Loss → Gradient → Update only unfrozen layers
```

📌 Frozen layer গুলো gradient পেলেও **apply হয় না**

---

# 🔹 Example: Weight freeze vs trainable

```python
for layer in model.layers:
    print(layer.name, layer.trainable)
```

👉 Output:

```
conv1 False
conv2 False
dense1 True
dense2 True
```

---

# 🧪 Common Use-Cases

* Image classification (ResNet, VGG, MobileNet)
* NLP (BERT embeddings)
* Speech models
* Small dataset training

---

# ❌ Common Mistake

❌ Freeze করার পরে আবার compile না করা

✔ Correct:

```python
model.compile(...)
```

📌 trainable change করলে **compile আবার করতে হবে**

---

# 🧠 Interview-ready One-liners

* **Weight freeze** means stopping gradient updates for selected layers
* Used in **transfer learning** to preserve pretrained knowledge
* Improves generalization on small datasets

---

# ✅ TL;DR (Short Summary)

| Question              | Answer                          |
| --------------------- | ------------------------------- |
| Weight freeze মানে?   | Weight update বন্ধ              |
| কেন লাগে?             | Pretrained knowledge রাখার জন্য |
| কোথায় ব্যবহার?        | Transfer learning               |
| trainable=False করলে? | Layer freeze                    |
| Compile দরকার?        | হ্যাঁ, আবার                     |

---


