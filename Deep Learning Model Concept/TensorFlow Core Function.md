

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

ঠিক আছে 👍
এবার আমি আগের ডকুমেন্টেশনটা **আরও গভীরে গিয়ে** ব্যাখ্যা করছি এবং **প্রতিটা function-এর জন্য আলাদা আলাদা code example** দিচ্ছি, যাতে তুমি—

* 🔹 concept বুঝো
* 🔹 exam/viva-তে explain করতে পারো
* 🔹 নিজে code লিখতে পারো (without confusion)

আমি এটাকে **step-by-step pipeline + function-wise deep explanation** হিসেবে সাজালাম।

---

# 📘 Keras Deep Learning Functions

## (More Explanation + Code Examples)

---

# 🔹 1. `mnist.load_data()`

## 👉 কী কাজ করে?

* MNIST dataset **download + load** করে
* Dataset already **train/test split করা থাকে**

## 👉 কেন দরকার?

* Data ছাড়া model train করা যাবে না
* Beginner-দের জন্য built-in dataset

---

## ✅ Syntax

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## 🔸 Parameters

```
কোনো parameter লাগে না
```

## 🔁 Returns (খুব গুরুত্বপূর্ণ)

* 2টা tuple return করে
  1️⃣ Training data
  2️⃣ Testing data

---

## ✅ Code Example

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
```

📌 এখানে:

* `x_train` → image
* `y_train` → label (0–9)

---

# 🔹 2. `Sequential()`

## 👉 কী কাজ করে?

* Neural network model তৈরি করে
* Layers গুলোকে **একটার পর একটা stack করে**

## 👉 কখন ব্যবহার করবে?

* Single input
* Single output
* No branching / skip connection

---

## ✅ Syntax

```python
model = Sequential(layers)
```

## 🔸 Parameter

| Name   | Meaning       |
| ------ | ------------- |
| layers | layer এর list |

## 🔁 Returns

```
model object
```

---

## ✅ Code Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1)
])
```

📌 এখন `model` object দিয়ে সব কাজ হবে

---

# 🔹 3. `Dense()`

## 👉 কী কাজ করে?

* Fully Connected Layer
* Input feature গুলো combine করে output দেয়

---

## ✅ Syntax

```python
Dense(units, activation=None)
```

## 🔸 Parameters

| Parameter  | Mandatory | Example |
| ---------- | --------- | ------- |
| units      | ✅         | 32      |
| activation | ❌         | 'relu'  |

---

## 🔁 Returns

```
Dense layer object
```

---

## ✅ কোন ক্ষেত্রে কোন Dense?

### 🔹 Hidden layer

```python
Dense(64, activation='relu')
```

### 🔹 Binary classification

```python
Dense(1, activation='sigmoid')
```

### 🔹 Multi-class classification

```python
Dense(10, activation='softmax')
```

---

# 🔹 4. `Flatten()`

## 👉 কী কাজ করে?

* Image বা multi-dimensional data → 1D vector বানায়

---

## ✅ Syntax

```python
Flatten()
```

## 🔸 Parameters

```
None
```

## 🔁 Returns

```
Flatten layer object
```

---

## ✅ Code Example

```python
from tensorflow.keras.layers import Flatten

# 28x28 image → 784 vector
Flatten(input_shape=(28,28))
```

📌 Dense layer image directly নিতে পারে না, তাই Flatten দরকার

---

# 🔹 5. `model.compile()`

## 👉 কী কাজ করে?

* Model-কে **training-ready** করে
* বলে দেয়:

  * কীভাবে weight update হবে
  * error কীভাবে calculate হবে

---

## ✅ Syntax (Minimal)

```python
model.compile(optimizer, loss)
```

## 🔸 Mandatory Parameters

| Parameter | Meaning            |
| --------- | ------------------ |
| optimizer | weight update rule |
| loss      | error function     |

---

## ✅ Code Examples

### 🔹 MNIST (integer labels)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)
```

### 🔹 Binary classification

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
```

---

## 🔁 Returns

```
None
```

📌 কিন্তু internal configuration তৈরি হয়

---



---

# 📘 `model.compile()` – Full Detailed Documentation

---

## 🔹 1. `model.compile()` কী?

👉 `compile()` হলো **model training-এর আগে বাধ্যতামূলক ধাপ**
👉 এটা model-কে বলে দেয়:

1️⃣ **কীভাবে weight update হবে** → optimizer
2️⃣ **ভুল (error) কীভাবে মাপা হবে** → loss
3️⃣ **performance কীভাবে দেখানো হবে** → metrics (optional)

📌 সহজ ভাষায়:

> **`compile()` = training rules set করা**

---

## 🔹 2. Minimal Syntax (সবচেয়ে ছোট form)

```python
model.compile(optimizer, loss)
```

📌 এটুকু দিলেই model train করা যাবে

---

## 🔹 3. Mandatory Parameters ⭐

### ✅ (1) `optimizer`

#### 👉 কী কাজ করে?

* Loss কমানোর জন্য **weights কীভাবে change হবে** সেটা ঠিক করে
* Gradient descent-এর strategy

---

#### 🔸 Common Optimizers

| Optimizer | কখন ব্যবহার                   |
| --------- | ----------------------------- |
| `sgd`     | Basic learning                |
| `adam` ⭐  | Most popular (default choice) |
| `rmsprop` | RNN / noisy data              |

---

#### ✅ Minimal Example

```python
optimizer='adam'
```

📌 Beginner + MNIST + most DL problem-এ **adam best**

---

### ✅ (2) `loss`

#### 👉 কী কাজ করে?

* Model prediction আর **true label-এর পার্থক্য** মাপে
* Backpropagation এই loss দিয়েই হয়

---

## 🔹 4. Loss Function Selection (Very Important ⭐)

### 🔹 Case-wise Loss Table

| Problem Type                | Output Layer      | Loss Function                   |
| --------------------------- | ----------------- | ------------------------------- |
| Regression                  | Dense(1)          | mean_squared_error              |
| Binary classification       | Dense(1, sigmoid) | binary_crossentropy             |
| Multi-class (integer label) | Dense(n, softmax) | sparse_categorical_crossentropy |
| Multi-class (one-hot label) | Dense(n, softmax) | categorical_crossentropy        |

---

### ✅ MNIST (integer labels: 0–9)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)
```

📌 কারণ:

* Labels → integer (0,1,2…9)
* One-hot encoding করা হয়নি

---

### ✅ Binary Classification

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
```

📌 Output:

```python
Dense(1, activation='sigmoid')
```

---

### ✅ Regression Example

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)
```

---

## 🔹 5. Optional Parameter: `metrics` (Monitoring Only)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

📌 মনে রাখবে:

* **metrics training change করে না**
* শুধু performance দেখায়

---

## 🔹 6. `compile()` internally কী করে? (Concept)

`compile()` করার সময় model:

1️⃣ Optimizer object তৈরি করে
2️⃣ Loss function attach করে
3️⃣ Metrics tracker attach করে
4️⃣ Training graph প্রস্তুত করে

📌 এই ধাপ ছাড়া `fit()` জানে না:

* কী optimize করবে
* কী minimize করবে

---

## 🔹 7. `compile()` কী return করে?

```
None
```

❗ কিন্তু:

* Model object-এর ভিতরে **internal state তৈরি হয়**

---

## 🔹 8. কেন `compile()` ছাড়া `fit()` কাজ করে না?

❌ Wrong

```python
model.fit(x_train, y_train, epochs=5)
```

📌 Error:

```
You must compile your model before training.
```

কারণ:

* Optimizer নেই
* Loss নেই
* Training rule undefined

---

## 🔹 9. Minimal End-to-End Example

```python
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

model.fit(x_train, y_train, epochs=5)
```

---

## 🔹 10. Common Mistakes ❌ (Exam Favorite)

### ❌ Wrong loss for integer labels

```python
loss='categorical_crossentropy'  # WRONG
```

✔ Correct:

```python
loss='sparse_categorical_crossentropy'
```

---

### ❌ Regression-এ accuracy

```python
metrics=['accuracy']  # WRONG
```

---

### ❌ Binary output কিন্তু softmax loss

```python
Dense(1, activation='sigmoid')
loss='categorical_crossentropy'  # WRONG
```

---

## 🔹 11. `compile()` vs `fit()` vs `evaluate()`

| Function | Role               |
| -------- | ------------------ |
| compile  | Training rules set |
| fit      | Model শেখে         |
| evaluate | Model test হয়      |

---

## 🧠 Exam / Viva One-Liners ⭐

* **`compile()` makes model training-ready**
* **optimizer controls weight update**
* **loss controls learning**
* **compile before fit is mandatory**

---

## ✅ Final Takeaway (Golden Line)

> 🔹 **`model.compile()` defines HOW the model will learn**
> 🔹 Without compile → no training possible

---


# 🔹 6. `model.fit()`

## 👉 কী কাজ করে?

* Model training শুরু করে
* Data দেখে weight শিখে

---

## ✅ Syntax (Minimal)

```python
model.fit(x, y, epochs)
```

## 🔸 Parameters

| Name   | Meaning             |
| ------ | ------------------- |
| x      | training data       |
| y      | training labels     |
| epochs | কতবার dataset দেখবে |

---

## 🔁 Returns

```
History object
```

---

## ✅ Code Example

```python
history = model.fit(x_train, y_train, epochs=5)

print(history.history.keys())
```

📌 loss, accuracy track করা যায়

---

---

# 📘 `model.fit()` – Full Detailed Documentation

---

## 🔹 1. `model.fit()` কী কাজ করে?

👉 Neural network-কে **train** করে
👉 Data দেখিয়ে **weights update** করে
👉 Loss কমানোর চেষ্টা করে

📌 সহজ কথায়:

> **`fit()` = model শেখে**

---

## 🔹 2. Minimal Syntax (সবচেয়ে ছোট form)

```python
history = model.fit(x, y, epochs)
```

এটাই **সবচেয়ে minimum working call**।

---

## 🔹 3. Minimal Required Parameters ⭐

### ✅ `x` — Training Data

| বিষয়    | Explanation                  |
| ------- | ---------------------------- |
| Type    | numpy array / tensor         |
| Meaning | Input features               |
| Shape   | `(num_samples, input_shape)` |

#### Example (MNIST)

```python
x_train.shape = (60000, 28, 28)
```

---

### ✅ `y` — Training Labels

| বিষয়    | Explanation                                      |
| ------- | ------------------------------------------------ |
| Type    | numpy array / tensor                             |
| Meaning | True output                                      |
| Shape   | `(num_samples,)` বা `(num_samples, num_classes)` |

#### Example

```python
y_train.shape = (60000,)
```

📌 `x` আর `y`-র **first dimension একই হতে হবে**

---

### ✅ `epochs` — Training Loop Count

| বিষয়    | Explanation                    |
| ------- | ------------------------------ |
| Type    | int                            |
| Meaning | Dataset কয়বার পুরোটা train হবে |
| Example | `epochs=5`                     |

📌 `epochs=1` মানে:

> পুরো training data একবার দেখা

---

## 🔹 4. Parameter না দিলে কী হবে?

❌ Wrong

```python
model.fit(x_train, y_train)
```

📌 Error হবে, কারণ:

* `epochs` mandatory

---

## 🔹 5. `model.fit()` internally কী করে? (Step-by-Step)

প্রতিটা epoch-এ 👇

1️⃣ `x` নেয়
2️⃣ Forward pass করে
3️⃣ Prediction বের করে
4️⃣ Loss calculate করে
5️⃣ Backpropagation
6️⃣ Weight update
7️⃣ Metric calculate করে

📌 এই cycle **epochs বার repeat হয়**

---

## 🔹 6. `model.fit()` কী return করে? ⭐

### 🔁 Return Type

```
History object
```

---

## 🔹 7. `History object` কী?

👉 Training-এর সময় **সব metric record করে**
👉 Python object আকারে থাকে

---

### 🔍 Structure

```python
history.history
```

এটা একটা dictionary 👇

```python
{
  'loss': [...],
  'accuracy': [...]
}
```

---

### ✅ Code Example

```python
history = model.fit(x_train, y_train, epochs=5)

print(history.history.keys())
```

Output:

```
dict_keys(['loss', 'accuracy'])
```

---

### 🔹 Epoch-wise value access

```python
print(history.history['loss'])
print(history.history['accuracy'])
```

📌 Plot করার জন্য ব্যবহার হয়

---

## 🔹 8. Metrics থাকলে কী হয়?

### Compile:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Fit output:

```
Epoch 1/5
loss: 0.42 - accuracy: 0.88
```

📌 `metrics` না দিলে শুধু loss দেখাবে

---

## 🔹 9. Common Optional Parameters (Understanding Purpose)

(Exam-এ না এলেও concept জানা দরকার)

| Parameter       | Purpose                |
| --------------- | ---------------------- |
| batch_size      | একসাথে কয়টা sample     |
| validation_data | validation performance |
| verbose         | output style           |

📌 কিন্তু **minimum training-এর জন্য এগুলো দরকার নেই**

---

## 🔹 10. Very Common Beginner Mistakes ❌

### ❌ x, y shape mismatch

```python
x.shape = (1000, 28, 28)
y.shape = (900,)
```

---

### ❌ Fit before compile

```python
model.fit(...)   # ERROR
```

📌 `compile()` mandatory

---

### ❌ Expecting prediction from fit

```python
y_pred = model.fit(...)  # WRONG
```

📌 Prediction → `predict()`

---

## 🔹 11. `fit()` vs `evaluate()` vs `predict()`

| Function | Learns | Needs y | Returns       |
| -------- | ------ | ------- | ------------- |
| fit      | ✅      | ✅       | History       |
| evaluate | ❌      | ✅       | loss, metrics |
| predict  | ❌      | ❌       | predictions   |

---

## 🔹 12. Minimal End-to-End Example

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train, y_train, epochs=5)
```

---

## 🧠 Exam / Viva One-Liners ⭐

* **`model.fit()` trains the model**
* **epochs defines training repetitions**
* **returns History object**
* **compile() before fit() is mandatory**

---

## ✅ Final Takeaway

* `model.fit()`-এর **minimum parameters = x, y, epochs**
* Training-এর সব তথ্য **History object-এ থাকে**
* Learning only happens here

---


# 🔹 7. `model.evaluate()`

## 👉 কী কাজ করে?

* Trained model কতটা ভালো কাজ করছে সেটা মাপে

---

## ✅ Syntax

```python
model.evaluate(x, y)
```

## 🔸 Parameters

| Name | Meaning     |
| ---- | ----------- |
| x    | test data   |
| y    | true labels |

---

## 🔁 Returns

```
loss, metrics
```

---

## ✅ Code Example

```python
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc)
```


---

# 📘 `model.evaluate()` – Full Detailed Explanation (Minimal Focus)

---

## 🔹 1. `model.evaluate()` কী কাজ করে?

👉 Training শেষ হওয়ার পর
👉 Model শেখা weight ব্যবহার করে
👉 **Test / validation data-তে model-এর performance মাপে**

📌 এটা **training করে না**
📌 এটা **prediction দেয় না**
📌 শুধু **loss + metric calculate করে**

---

## 🔹 2. Minimal Syntax (সবচেয়ে ছোট form)

```python
result = model.evaluate(x, y)
```

বা (সবচেয়ে common)

```python
loss, metric = model.evaluate(x, y)
```

---

## 🔹 3. Minimal Required Parameters ⭐

### ✅ `x` (MANDATORY)

| বিষয়           | Explanation                        |
| -------------- | ---------------------------------- |
| Parameter name | `x`                                |
| Type           | numpy array / tensor               |
| Meaning        | Test / validation input data       |
| Shape          | `(number_of_samples, input_shape)` |

📌 `x` = input data
📌 Training data নয়, সাধারণত **test data**

---

### ✅ `y` (MANDATORY)

| বিষয়           | Explanation                                                  |
| -------------- | ------------------------------------------------------------ |
| Parameter name | `y`                                                          |
| Type           | numpy array / tensor                                         |
| Meaning        | True labels (ground truth)                                   |
| Shape          | `(number_of_samples,)` বা `(number_of_samples, num_classes)` |

📌 `y` ছাড়া loss calculate করা সম্ভব না

---

## 🔹 4. Parameter না দিলে কী হবে?

❌ Wrong

```python
model.evaluate(x_test)
```

📌 Error আসবে, কারণ:

* loss function কে true label দরকার

---

## 🔹 5. `model.evaluate()` কী return করে? ⭐

### 🔁 Return Type

```
float অথবা list of floats
```

Return format **depend করে** `compile()`-এ কী দিয়েছ তার উপর।

---

## 🔹 6. Compile অনুযায়ী Return Value (Very Important)

---

### 🔹 Case 1: Only loss দেওয়া আছে

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)
```

### 🔁 Return

```python
loss = model.evaluate(x_test, y_test)
```

📌 Single float return করে
📌 Example:

```
0.2456
```

---

### 🔹 Case 2: Loss + One Metric

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 🔁 Return

```python
loss, accuracy = model.evaluate(x_test, y_test)
```

📌 Two values return করে
📌 Example:

```
loss = 0.24
accuracy = 0.92
```

---

### 🔹 Case 3: Multiple Metrics

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision']
)
```

### 🔁 Return

```python
loss, acc, prec = model.evaluate(x_test, y_test)
```

📌 Return order:

```
[loss, metric1, metric2, ...]
```

---

## 🔹 7. Return Value-এর Shape / Meaning

| Return value       | Meaning                       |
| ------------------ | ----------------------------- |
| loss               | Model-এর average error        |
| accuracy           | Correct prediction percentage |
| precision / recall | Advanced metric               |

📌 সব value **test dataset-এর উপর average**

---

## 🔹 8. Full Minimal Code Example (MNIST)

```python
loss, acc = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", acc)
```

---

## 🔹 9. Difference: `evaluate()` vs `predict()`

| Feature            | evaluate() | predict() |
| ------------------ | ---------- | --------- |
| Needs true label   | ✅          | ❌         |
| Returns loss       | ✅          | ❌         |
| Returns prediction | ❌          | ✅         |
| Training           | ❌          | ❌         |

---

## 🔹 10. Very Common Beginner Confusions ❌

### ❌ Using training data in evaluate

```python
model.evaluate(x_train, y_train)  # WRONG practice
```

📌 Test data ব্যবহার করা উচিত

---

### ❌ Expecting class labels

```python
loss, y_pred = model.evaluate(x_test, y_test)  # WRONG
```

📌 `evaluate()` prediction দেয় না

---

## 🔹 11. Exam / Viva One-Liners ⭐

* **`model.evaluate()` needs both input data and true labels**
* **It returns loss and metrics, not predictions**
* **It is used for testing, not training**

---

## ✅ Final Takeaway

* `model.evaluate()`-এর **minimum parameters = x, y**
* Return value নির্ভর করে **compile() এ দেওয়া metrics-এর উপর**
* Model performance measure করার একমাত্র standard method

---


---

# 🔹 8. `model.predict()`

## 👉 কী কাজ করে?

* New data দিয়ে prediction দেয়

---

## ✅ Syntax

```python
model.predict(x)
```

## 🔁 Returns

* Regression → value
* Binary → probability
* Multi-class → probability vector

---

## ✅ Code Example

```python
import numpy as np

pred = model.predict(x_test)

print(pred[0])              # probabilities
print(np.argmax(pred[0]))   # predicted class
```




---

# 📘 `model.predict()` – Full Detailed Explanation (Minimal Focus)

---

## 🔹 1. `model.predict()` কী কাজ করে?

👉 Training শেষ হওয়ার পর
👉 Model শেখা weight ব্যবহার করে
👉 **নতুন / অজানা data-এর output অনুমান (predict)** করে

📌 `predict()` **training করে না**, শুধু inference করে।

---

## 🔹 2. Minimal Syntax (সবচেয়ে ছোট form)

```python
pred = model.predict(x)
```

---

## 🔹 3. Minimal Required Parameter ⭐

### ✅ `x` (MANDATORY)

| বিষয়           | Explanation                        |
| -------------- | ---------------------------------- |
| Parameter name | `x`                                |
| Type           | numpy array / tensor               |
| Meaning        | Input data                         |
| Shape          | `(number_of_samples, input_shape)` |

📌 **`x` ছাড়া `predict()` কাজ করবে না**

---

### 🔍 `x` কী value নিতে পারে?

| Model input   | `x` shape          |
| ------------- | ------------------ |
| Tabular       | `(N, features)`    |
| MNIST image   | `(N, 28, 28)`      |
| CNN image     | `(N, H, W, C)`     |
| Single sample | `(1, input_shape)` |

---

## 🔹 4. `predict()` কী return করে? ⭐ (Very Important)

### 🔁 Return Type

```
numpy.ndarray
```

---

## 🔹 5. Task-wise Return Value (Detail)

---

### 🔹 Case 1: Regression Model

```python
Dense(1, activation='linear')
```

### 🔁 Return

```
shape = (N, 1)
```

Example:

```python
[[23.5],
 [18.2]]
```

📌 Predicted continuous value

---

### 🔹 Case 2: Binary Classification

```python
Dense(1, activation='sigmoid')
```

### 🔁 Return

```
shape = (N, 1)
```

Example:

```python
[[0.87],
 [0.12]]
```

📌 Probability
📌 Class rule:

```python
prob > 0.5 → class 1
```

---

### 🔹 Case 3: Multi-Class Classification (MNIST)

```python
Dense(10, activation='softmax')
```

### 🔁 Return

```
shape = (N, 10)
```

Example:

```python
[0.01, 0.02, 0.90, 0.01, ...]
```

📌 Each value = class probability
📌 Sum = 1

---

## 🔹 6. Single Sample Prediction (Very Common Confusion)

### ❌ Wrong

```python
model.predict(x_test[0])   # shape mismatch
```

### ✅ Correct

```python
model.predict(x_test[0:1])
```

📌 Reason:

* Model expects **batch dimension**
* Shape must be `(1, 28, 28)`

---

## 🔹 7. Converting Prediction → Class Label

### Multi-class

```python
pred = model.predict(x)
predicted_class = np.argmax(pred, axis=1)
```

---

### Binary

```python
pred = model.predict(x)
predicted_class = (pred > 0.5).astype(int)
```

---

## 🔹 8. Full Minimal Example (MNIST)

```python
pred = model.predict(x_test)

print(pred.shape)        # (10000, 10)
print(pred[0])           # probabilities
print(np.argmax(pred[0]))  # predicted digit
```

---

## 🔹 9. Summary Table (Exam Gold ⭐)

| Aspect            | Detail               |
| ----------------- | -------------------- |
| Function          | `model.predict()`    |
| Minimal parameter | `x`                  |
| Parameter type    | numpy array / tensor |
| Returns           | numpy array          |
| Regression        | value                |
| Binary            | probability          |
| Multi-class       | probability vector   |

---

## 🔹 10. Viva / Exam One-Liner

> **`model.predict()` takes only input data and returns model output without training.**

---

## ✅ Final Takeaway

* `predict()`-এর **only mandatory parameter = input data**
* Output সবসময় **array আকারে আসে**
* Class বের করতে **post-processing** (argmax / threshold) লাগে
* Training আর prediction আলাদা ধাপ

---







---

# 🔹 9. `model.summary()`

## 👉 কী কাজ করে?

* Model architecture table আকারে দেখায়

---

## ✅ Syntax

```python
model.summary()
```

## 🔁 Returns

```
None (prints output)
```

---

## 🧠 Full Minimal Pipeline Example (All Together)

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
model.predict(x_test)
```

---

## ✅ Final Big Picture (Exam Line)

> **Deep Learning workflow:**
> Data → Model → Compile → Fit → Evaluate → Predict

---


