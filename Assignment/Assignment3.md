

1️⃣ ( y = 5x + 10 )
2️⃣ ( y = 3x^2 + 5x + 10 )
3️⃣ ( y = 4x^3 + 3x^2 + 5x + 10 )

প্রতিটা কোডে থাকবে:

* ✅ Dataset preparation (Train / Validation / Test)
* ✅ FCFNN architecture
* ✅ EarlyStopping
* ✅ Training
* ✅ Loss curve
* ✅ Original vs Predicted plot
* ✅ MSE evaluation

---

# 🔵 CODE 1: Linear Equation

# y = 5x + 10

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Generate Data
x = np.linspace(-10, 10, 1000)
y = 5*x + 10

x = x.reshape(-1,1)
y = y.reshape(-1,1)

# Split Data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

# Build Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate
loss, mae = model.evaluate(x_test, y_test)
print("Test MSE:", loss)

# Plot Loss Curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve (Linear)")
plt.legend()
plt.show()

# Plot Prediction
y_pred = model.predict(x_test)
plt.scatter(x_test, y_test, label='Original')
plt.scatter(x_test, y_pred, label='Predicted')
plt.legend()
plt.title("Original vs Predicted (Linear)")
plt.show()
```

---

# 🟡 CODE 2: Quadratic Equation

# y = 3x² + 5x + 10

```python
# Generate Data
x = np.linspace(-10, 10, 2000)
y = 3*(x**2) + 5*x + 10

x = x.reshape(-1,1)
y = y.reshape(-1,1)

# Split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

# Build Model (Slightly Bigger)
model = Sequential([
    Dense(32, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(patience=15, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    callbacks=[early_stop],
    verbose=0
)

loss, mae = model.evaluate(x_test, y_test)
print("Test MSE:", loss)

# Plot Loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss Curve (Quadratic)")
plt.legend()
plt.show()

# Plot Prediction
y_pred = model.predict(x_test)
plt.scatter(x_test, y_test, label='Original')
plt.scatter(x_test, y_pred, label='Predicted')
plt.legend()
plt.title("Original vs Predicted (Quadratic)")
plt.show()
```

---

# 🔴 CODE 3: Cubic Equation

# y = 4x³ + 3x² + 5x + 10

```python
# Generate Data
x = np.linspace(-10, 10, 4000)
y = 4*(x**3) + 3*(x**2) + 5*x + 10

x = x.reshape(-1,1)
y = y.reshape(-1,1)

# Split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

# Build Model (Deeper)
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=500,
    callbacks=[early_stop],
    verbose=0
)

loss, mae = model.evaluate(x_test, y_test)
print("Test MSE:", loss)

# Plot Loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss Curve (Cubic)")
plt.legend()
plt.show()

# Plot Prediction
y_pred = model.predict(x_test)
plt.scatter(x_test, y_test, label='Original')
plt.scatter(x_test, y_pred, label='Predicted')
plt.legend()
plt.title("Original vs Predicted (Cubic)")
plt.show()
```

---

# 🧠 Effect of Power on FCFNN Architecture

| Equation Type  | Complexity | Required Neurons | Required Data  |
| -------------- | ---------- | ---------------- | -------------- |
| Linear (x¹)    | Low        | Few neurons      | Less data      |
| Quadratic (x²) | Medium     | More neurons     | More data      |
| Cubic (x³)     | High       | Deep network     | Much more data |

---

## 🔎 Why?

Higher power means:

* More non-linearity
* More curvature
* More complex mapping
* Need deeper network
* Need more training samples

---

# 🎯 Final Concept Summary

* Power increases → Model complexity increases
* Complexity increases → Need more layers & neurons
* Higher order polynomial → More training data required

---

