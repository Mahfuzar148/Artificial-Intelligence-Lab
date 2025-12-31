

---

# 📘 FULL DOCUMENTATION: Fully Connected Neural Network (FCNN)

---

## 🔹 1. FCNN কী?

**Fully Connected Neural Network (FCNN)** হলো এমন neural network যেখানে:

* প্রতিটি neuron আগের layer-এর **সব neuron-এর সাথে connected**
* `Dense` layer দিয়ে তৈরি
* Numerical, tabular, flattened image data-তে বেশি ব্যবহার হয়

📌 CNN ছাড়া image দিলে সেটাও FCNN-ই হয় (Flatten ব্যবহার করলে)

---

## 🔹 2. FCNN-এর Common General Format

### 🔧 Mathematical View

```
y = f(Wx + b)
```

### 🔧 Layer-wise View

```
Input
 ↓
Dense + Activation
 ↓
Dense + Activation
 ↓
Output Layer
```

---

## 🔹 3. Import Section (From Your Pic)

```python
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
```

---

# 🧪 EXAMPLES 
---

## ✅ Example 1: Simplest FCNN (NO Activation)

📌 *Build the simplest fully connected neural network without activation function*

```python
inputs = Input((1,))
outputs = Dense(1, name='OutputLayer')(inputs)
model = Model(inputs, outputs)
model.summary()
```

### 🔍 Explanation

* Input dimension = 1
* Output neuron = 1
* No activation → **Linear model**
* Used in **simple regression**

### 📊 Parameters

```
Params = (1 weight + 1 bias) = 2
```

---

## ✅ Example 2: Simplest FCNN WITH Activation (Inline)

📌 *Build the simplest FCNN with activation function*

```python
inputs = Input((1,))
outputs = Dense(1, activation='sigmoid', name='OutputLayer')(inputs)
model = Model(inputs, outputs, name='FCNN_with_Activation')
model.summary()
```

### 🔍 Explanation

* Sigmoid activation → output range (0,1)
* Used for **binary classification**
* Activation embedded inside Dense

---

## ✅ Example 3: FCNN with Separate Activation Layer

📌 *Build the simplest FCNN with separate layer for activation function*

```python
inputs = Input((1,))
x = Dense(1, name='OutputLayer')(inputs)
outputs = Activation('sigmoid', name='sigmoid')(x)
model = Model(inputs, outputs, name='FCNN_with_Activation')
model.summary()
```

### 🔍 Explanation

* Dense and Activation আলাদা
* Architecture আরও readable
* Teaching / research এ preferred

📌 Activation layer-এর **trainable parameter নেই**

---

## ✅ Example 4: Shallow FCNN (One Hidden Layer)

📌 *Build a simple shallow FCNN*

```python
inputs = Input((1,))
x = Dense(1, activation='sigmoid')(inputs)
outputs = Dense(1, activation='sigmoid', name='OutputLayer')(x)
model = Model(inputs, outputs, name='ShallowNN')
model.summary()
```

### 🔍 Explanation

* 1 hidden layer → **Shallow Network**
* Non-linearity introduce করে
* XOR type problem solve করতে পারে

---

## ✅ Example 5: Deep FCNN (DNN)

📌 *Build a deep FCNN*

```python
inputs = Input((1,))
x = Dense(2, activation='sigmoid')(inputs)
x = Dense(4, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
x = Dense(16, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
x = Dense(4, activation='sigmoid')(x)
outputs = Dense(1, activation='sigmoid', name='OutputLayer')(x)

model = Model(inputs, outputs, name='DNN')
model.summary()
```

### 🔍 Explanation

* Multiple hidden layers → **Deep Neural Network**
* Feature hierarchy শেখে
* Complex pattern modelling

---

## ✅ Example 6: Deep FCNN for Gray-Scale Image Data

📌 *Build Deep FCNN for gray-scale image data*

```python
inputs = Input((28, 28, 1))
x = Flatten()(inputs)

x = Dense(2, activation='sigmoid')(x)
x = Dense(4, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
x = Dense(16, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
x = Dense(4, activation='sigmoid')(x)

outputs = Dense(1, activation='sigmoid', name='OutputLayer')(x)

model = Model(inputs, outputs, name='DNN')
model.summary(show_trainable=True)
```

### 🔍 Explanation

* Image size = 28×28×1
* Flatten → 784 features
* CNN ছাড়া image classification (educational purpose)
* Parameter অনেক বেশি হয়

---

## ✅ Example 7: Deep FCNN as 3-Class Classifier (Gray Image)

📌 *Build a deep FCNN as a three-class classifier having grayscale input image*

```python
num_classes = 3

inputs = Input((28, 28, 1))
x = Flatten()(inputs)

x = Dense(2, activation='sigmoid')(x)
x = Dense(4, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
x = Dense(16, activation='sigmoid')(x)
x = Dense(8, activation='sigmoid')(x)
x = Dense(4, activation='sigmoid')(x)

outputs = Dense(num_classes, activation='softmax', name='OutputLayer')(x)

model = Model(inputs, outputs, name='DNN')
model.summary(show_trainable=True)
```

### 🔍 Explanation

* Softmax → multi-class probability
* Output neurons = number of classes
* Loss: `categorical_crossentropy`

---

## ✅ Example 8: 

📌 *Build a deep FCNN as a 10-class classifier for RGB input images*

---

## 📝 Homework Solution (Complete)

```python
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

num_classes = 10

inputs = Input((32, 32, 3))   # RGB image
x = Flatten()(inputs)

x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

outputs = Dense(num_classes, activation='softmax', name='OutputLayer')(x)

model = Model(inputs, outputs, name='FCNN_RGB_10_Class')
model.summary()
```

### 🔍 Explanation

* RGB → 3 channels
* Flatten mandatory for FCNN
* ReLU → faster convergence
* Softmax → multi-class classification

---

# 🧠 FCNN QUICK REFERENCE TABLE

| Task                       | Activation     |
| -------------------------- | -------------- |
| Regression                 | None           |
| Binary Classification      | Sigmoid        |
| Multi-class Classification | Softmax        |
| Hidden Layers              | ReLU / Sigmoid |

---

## ✅ Final Conclusion

* FCNN = Dense based network
* Deep FCNN = DNN
* Activation = non-linearity
* Flatten = image → FCNN bridge
* CNN is preferred for real image tasks

---

