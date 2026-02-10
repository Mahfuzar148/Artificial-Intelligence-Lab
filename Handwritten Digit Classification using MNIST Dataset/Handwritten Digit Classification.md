# Topic: Handwritten Digit Classification using MNIST Dataset with Keras

---

## 1. Introduction

This document explains step-by-step how to build, train, and evaluate a **multi-class classification model** using the **MNIST handwritten digit dataset**. The goal is to classify grayscale images of handwritten digits (0–9) using a **fully connected neural network (Dense layers)** implemented with **TensorFlow / Keras**.

Each code block used in the notebook is included below along with a clear explanation of what it does.

---

## 2. Import Necessary Modules

```python
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
```

### Explanation:

* `load_data`: Loads the MNIST dataset directly from Keras.
* `matplotlib.pyplot`: Used for visualizing images.
* `numpy`: Used for numerical operations.
* `to_categorical`: Converts class labels into one-hot encoded format.
* `Input, Flatten, Dense`: Layers used to build the neural network.
* `Model`: Used to create a functional Keras model.

---

## 3. Function to Display Images

```python
def display_img(img_set, title_set):
    n = len(title_set)
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(title_set[i])
    plt.show()
    plt.close()
```

### Explanation:

* This function displays images in a **3×3 grid**.
* `img_set`: Image data to display.
* `title_set`: Corresponding labels.
* `imshow`: Displays grayscale images.
* `plt.show()`: Renders the plot.

---

## 4. Load the MNIST Dataset

```python
(trainX, trainY), (testX, testY) = load_data()
```

### Explanation:

* `trainX`: Training images (60,000 samples).
* `trainY`: Training labels.
* `testX`: Testing images (10,000 samples).
* `testY`: Testing labels.

Each image has a size of **28 × 28 pixels**.

---

## 5. Investigate Loaded Data

```python
print('trainX.shape: {}, trainY.shape: {}, testX.shape: {}, testY.shape: {}'.format(
    trainX.shape, trainY.shape, testX.shape, testY.shape))

print('trainX.dtype: {}, trainY.dtype: {}, testX.dtype: {}, testY.dtype: {}'.format(
    trainX.dtype, trainY.dtype, testX.dtype, testY.dtype))

print('trainX.Range: {} - {}, testX.Range: {} - {}'.format(
    trainX.max(), trainX.min(), testX.max(), testX.min()))
```

### Explanation:

* Displays shape, data type, and pixel value range.
* Pixel values range from **0 to 255**.

---

## 6. Display Sample Images

```python
display_img(trainX[:9], trainY[:9])
```

### Explanation:

* Displays the first **9 training images** with their labels.

---

## 7. Expand Image Dimensions

```python
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
```

### Explanation:

* Converts images from **(28, 28)** to **(28, 28, 1)**.
* Required because deep learning models expect **4D input**: `(samples, height, width, channels)`.

---

## 8. Verify Updated Image Data

```python
print('trainX.shape: {}, testX.shape: {}'.format(trainX.shape, testX.shape))
print('trainX.dtype: {}, testX.dtype: {}'.format(trainX.dtype, testX.dtype))
print('trainX.Range: {} - {}, testX.Range: {} - {}'.format(
    trainX.max(), trainX.min(), testX.max(), testX.min()))
```

### Explanation:

* Confirms correct reshaping and data integrity.

---

## 9. Convert Labels to One-Hot Encoding

```python
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)
```

### Explanation:

* Converts numeric labels into vectors of length 10.
* Required for **categorical crossentropy loss**.

Example:

* Digit `3` → `[0 0 0 1 0 0 0 0 0 0]`

---

## 10. Verify Encoded Labels

```python
print('trainY.shape: {}, testY.shape: {}'.format(trainY.shape, testY.shape))
print(trainY[:5])
```

### Explanation:

* Confirms correct one-hot encoding.

---

## 11. Build the Neural Network Model

```python
inputs = Input((28, 28, 1), name='InputLayer')
x = Flatten()(inputs)
x = Dense(2, activation='relu')(x)
x = Dense(4, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(4, activation='relu')(x)
outputs = Dense(10, activation='softmax', name='OutputLayer')(x)

model = Model(inputs, outputs, name='Multi-Class-Classifier')
model.summary()
```

### Explanation:

* `Flatten`: Converts image to 1D vector (784 neurons).
* Multiple `Dense` layers learn hierarchical features.
* `ReLU`: Non-linear activation function.
* `Softmax`: Outputs probability distribution over 10 classes.

---

## 12. Compile the Model

```python
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Explanation:

* `categorical_crossentropy`: Used for multi-class classification.
* `accuracy`: Tracks classification accuracy.

---

## 13. Train the Model

```python
model.fit(
    trainX,
    trainY,
    batch_size=32,
    validation_split=0.1,
    epochs=10
)
```

### Explanation:

* `batch_size=32`: Number of samples per gradient update.
* `validation_split=0.1`: Uses 10% of training data for validation.
* `epochs=10`: Model trains for 10 full passes over the data.

---

## 14. Evaluate Model Performance

```python
model.evaluate(testX, testY)
```

### Explanation:

* Evaluates the trained model on unseen test data.
* Outputs test loss and accuracy.

---

## 15. Make Predictions

```python
predictY = model.predict(testX)

print('Originally  Predicted')
print('=========  =========')
for i in range(10):
    print(np.argmax(testY[i]), '\t\t', np.argmax(predictY[i]))
```

### Explanation:

* `model.predict`: Returns probability scores.
* `argmax`: Selects the predicted digit.
* Displays comparison between actual and predicted labels.

---

## 16. Conclusion

This project demonstrates a complete **deep learning workflow**:

* Data loading and preprocessing
* Visualization
* Model building using Keras Functional API
* Training, evaluation, and prediction

Although this model uses only Dense layers, it successfully performs handwritten digit classification and serves as a strong foundation before moving to **Convolutional Neural Networks (CNNs)**.

---

## **Full Source Code**

```python
# ==============================
# 1. Import Necessary Libraries
# ==============================

from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model


# ==============================
# 2. Function to Display Images
# ==============================

def display_img(img_set, title_set):
    n = len(title_set)
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(title_set[i])
    plt.show()
    plt.close()


# ==============================
# 3. Load MNIST Dataset
# ==============================

(trainX, trainY), (testX, testY) = load_data()


# ==============================
# 4. Investigate Loaded Data
# ==============================

print('trainX.shape: {}, trainY.shape: {}, testX.shape: {}, testY.shape: {}'.format(
    trainX.shape, trainY.shape, testX.shape, testY.shape))

print('trainX.dtype: {}, trainY.dtype: {}, testX.dtype: {}, testY.dtype: {}'.format(
    trainX.dtype, trainY.dtype, testX.dtype, testY.dtype))

print('trainX.Range: {} - {}, testX.Range: {} - {}'.format(
    trainX.max(), trainX.min(), testX.max(), testX.min()))


# ==============================
# 5. Display Sample Images
# ==============================

display_img(trainX[:9], trainY[:9])


# ==============================
# 6. Expand Image Dimensions
# ==============================

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)


# ==============================
# 7. Verify Updated Image Data
# ==============================

print('trainX.shape: {}, testX.shape: {}'.format(trainX.shape, testX.shape))
print('trainX.dtype: {}, testX.dtype: {}'.format(trainX.dtype, testX.dtype))
print('trainX.Range: {} - {}, testX.Range: {} - {}'.format(
    trainX.max(), trainX.min(), testX.max(), testX.min()))


# ==============================
# 8. One-Hot Encode Labels
# ==============================

trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)


# ==============================
# 9. Verify Encoded Labels
# ==============================

print('trainY.shape: {}, testY.shape: {}'.format(trainY.shape, testY.shape))
print(trainY[:5])


# ==============================
# 10. Build the Neural Network
# ==============================

inputs = Input((28, 28, 1), name='InputLayer')

x = Flatten()(inputs)
x = Dense(2, activation='relu')(x)
x = Dense(4, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(4, activation='relu')(x)

outputs = Dense(10, activation='softmax', name='OutputLayer')(x)

model = Model(inputs, outputs, name='Multi-Class-Classifier')
model.summary()


# ==============================
# 11. Compile the Model
# ==============================

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ==============================
# 12. Train the Model
# ==============================

model.fit(
    trainX,
    trainY,
    batch_size=32,
    validation_split=0.1,
    epochs=10
)


# ==============================
# 13. Evaluate the Model
# ==============================

model.evaluate(testX, testY)


# ==============================
# 14. Make Predictions
# ==============================

predictY = model.predict(testX)

print('Originally  Predicted')
print('=========  =========')

for i in range(10):
    print(np.argmax(testY[i]), '\t\t', np.argmax(predictY[i]))
```

---

## ✅ Notes (Important for Exams / Viva)

* **Input shape**: `(28, 28, 1)`
* **Output layer**: 10 neurons (digits 0–9)
* **Loss function**: `categorical_crossentropy`
* **Activation**: `ReLU` (hidden layers), `Softmax` (output)
* **Model type**: Fully Connected Neural Network (not CNN)

---
### End of Document
