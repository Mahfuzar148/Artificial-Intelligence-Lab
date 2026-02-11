

---

# 🟢 PART 1: model.compile()

## 🔹 What is `model.compile()`?

`model.compile()` configures **how the model will learn**.

Think of it as:

🧠 **compile() = Set learning rules**

It defines:

* 🔵 Optimizer (How weights update)
* 🔴 Loss (How error is measured)
* 🟣 Metrics (What performance we track)

---

## ✅ Minimal Compile Example (Regression)

```python
model.compile(
    optimizer='adam',
    loss='mse'
)
```

✔ Used for regression problems
✔ No metrics needed (optional)

---

## 🟡 Binary Classification Example

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

✔ Output layer → `Dense(1, activation='sigmoid')`
✔ Labels → 0 / 1

---

## 🔵 Multi-class (One-hot Labels)

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

✔ Output → `Dense(num_classes, activation='softmax')`
✔ Labels → One-hot encoded

---

## 🟣 Multi-class (Integer Labels)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

✔ Labels → 0,1,2,3…
✔ No need for one-hot encoding

---

## 🟠 Advanced Optimizer Control

```python
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mse'
)
```

🧠 Used when:

* Training unstable
* Need fine control

---

## 🔴 Multi-output Model Example

```python
model.compile(
    optimizer='adam',
    loss=['mse', 'binary_crossentropy'],
    loss_weights=[0.7, 0.3]
)
```

✔ When model has multiple outputs
✔ One output more important

---

# 🟢 PART 2: model.fit()

## 🔹 What is `model.fit()`?

`model.fit()` actually trains the model.

🧠 **fit() = Execute learning process**

It performs:

* Forward pass
* Loss calculation
* Backpropagation
* Weight update

---

## ✅ Minimal Training Example

```python
model.fit(
    x_train,
    y_train,
    epochs=5
)
```

✔ Simplest possible training

---

## 🟡 With Batch Size

```python
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32
)
```

🧠 Small batch → stable but slower
🧠 Large batch → faster but memory heavy

---

## 🔵 With Validation Split

```python
model.fit(
    x_train,
    y_train,
    epochs=20,
    validation_split=0.2
)
```

✔ 20% data used for validation

---

## 🟣 With Separate Validation Data

```python
model.fit(
    x_train,
    y_train,
    epochs=20,
    validation_data=(x_val, y_val)
)
```

✔ When validation dataset already prepared

---

## 🔴 Early Stopping (Advanced)

```python
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(patience=3)

model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[callback]
)
```

✔ Stops training automatically

---

## 🟠 Class Weight (Imbalanced Data)

```python
model.fit(
    x_train,
    y_train,
    epochs=15,
    class_weight={0:1.0, 1:3.0}
)
```

✔ Used when one class appears less frequently

---

## 🟤 Sample Weight

```python
model.fit(
    x_train,
    y_train,
    epochs=10,
    sample_weight=weights_array
)
```

✔ Used when some samples are more important

---

## ⚫ Using Data Generator

```python
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=100
)
```

✔ Needed when dataset is too large for memory

---

# 🟢 Complete Case Examples

---

## 🟡 Regression Complete Example

```python
model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=32
)
```

---

## 🔵 Binary Classification Complete Example

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)
```

---

## 🟣 Multi-class Classification Complete Example

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val)
)
```

---

# 🎯 Final Concept Clarity

🧠 `model.compile()` → Defines learning rules
🧠 `model.fit()` → Executes learning process

---

# 📌 Quick Memory Trick

compile = Configure
fit = Train

---

End of Colorful Documentation 🎨
