
---

# 📚 **Fully Connected Neural Network (FCNN) **

---

## 1️⃣ **What is a Fully Connected Neural Network (FCNN)?**

A **Fully Connected Neural Network** (also called a **Dense Neural Network**) is a type of artificial neural network where:

* Every neuron in one layer is connected to **every neuron in the next layer**.
* Each connection has a **weight** that determines the strength of the signal.
* Each neuron has a **bias** that shifts the activation function output.
* An **activation function** is applied to introduce non-linearity so the network can learn complex patterns.

💡 **Key idea:** FCNN is the simplest and most common neural network structure, and it’s the building block for many advanced architectures.

---

### **Diagram of a Simple Neural Network**

![Simple Neural Network Diagram](https://github.com/Mahfuzar148/Artificial-Intelligence-Lab/blob/main/Assignment1/simple%20neural%20network.png)

---

## 2️⃣ **How Does It Work? (Step-by-Step)**

1. **Input Layer**
   Takes in feature values (e.g., $x_1, x_2$).

2. **Weighted Sum**
   Each neuron in the next layer calculates:

   $$
   z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + b
   $$

   where $w$ = weight, $b$ = bias.

3. **Activation Function**
   Applies a function like **ReLU**, **Sigmoid**, or **Softmax** to introduce non-linearity.

4. **Hidden Layers**
   Process features through multiple transformations to capture patterns.

5. **Output Layer**
   Produces final predictions — e.g., probability (classification) or a number (regression).

6. **Training**
   Uses **backpropagation** + **gradient descent** to adjust weights and biases to minimize error.

---

## 3️⃣ **When to Use FCNN?**

✅ **Good for:**

* Tabular data (structured datasets)
* Simple pattern recognition
* Problems where features are not sequential or spatial
* Small to medium-sized datasets

⚠️ **Not ideal for:**

* Image data → Convolutional Neural Networks (CNN) perform better
* Sequential data → Recurrent Neural Networks (RNN), Transformers are better
* Very large datasets → May overfit without regularization

---

## 4️⃣ **Why FCNN Can Perform Well**

* Learns **complex relationships** between inputs and outputs.
* Flexible — works with many types of problems.
* Easy to implement and train for smaller datasets.
* Fully connected structure ensures **all features are considered**.

---

## 5️⃣ **Python Implementation with Keras**

We’ll build the FCNN as in your diagram:

* **Input layer:** 2 neurons
* **Hidden layer 1:** 2 neurons (ReLU)
* **Hidden layer 2:** 3 neurons (ReLU)
* **Hidden layer 3:** 2 neurons (ReLU)
* **Output layer:** 1 neuron (Softmax in your code, but Sigmoid is better for binary classification)

---

### **Code**

```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# 1. Input Layer: 2 features
inputs = Input((2,))

# 2. First Hidden Layer: 2 neurons, ReLU activation
h1 = Dense(2, activation='relu')(inputs)

# 3. Second Hidden Layer: 3 neurons, ReLU activation
h2 = Dense(3, activation='relu')(h1)

# 4. Third Hidden Layer: 2 neurons, ReLU activation
h3 = Dense(2, activation='relu')(h2)

# 5. Output Layer: 1 neuron, Softmax activation
outputs = Dense(1, activation='softmax')(h3)

# Build the model
model = Model(inputs, outputs)

# Show model summary
model.summary()
```

---

### **Model Output Summary (Image)**

📷 **Model Summary Screenshot:**
[Model Summary Output](https://github.com/Mahfuzar148/Artificial-Intelligence-Lab/blob/main/Assignment1/simple%20neural%20network%20output.png)

---

## 6️⃣ **Code Explanation**

### **Layer 1 — Input**

* Shape `(2,)` → accepts 2 features $x_1, x_2$.
* No parameters yet (just input placeholders).

---

### **Layer 2 — Hidden Layer 1**

```python
h1 = Dense(2, activation='relu')(inputs)
```

* **2 neurons**, fully connected to input layer.
* **Parameters:** $(2 \text{ inputs} \times 2 \text{ neurons}) + 2 \text{ biases} = 6$.

---

### **Layer 3 — Hidden Layer 2**

```python
h2 = Dense(3, activation='relu')(h1)
```

* **3 neurons**, fully connected to H1.
* **Parameters:** $(2 \times 3) + 3 = 9$.

---

### **Layer 4 — Hidden Layer 3**

```python
h3 = Dense(2, activation='relu')(h2)
```

* **2 neurons**, fully connected to H2.
* **Parameters:** $(3 \times 2) + 2 = 8$.

---

### **Layer 5 — Output Layer**

```python
outputs = Dense(1, activation='softmax')(h3)
```

* **1 neuron** with softmax activation.
* **Parameters:** $(2 \times 1) + 1 = 3$.

---

### **Total Parameters**

$$
6 + 9 + 8 + 3 = 26
$$

---

## 7️⃣ **How This Maps to the Diagram**

Your diagram with $x_1, x_2$ → $h_1(2)$ → $h_2(3)$ → $h_3(2)$ → $y$ matches perfectly:

* **Every neuron is connected** to all neurons in the next layer.
* **w₁–w₁₈** = weights in connections.
* **b₁–b₇** = biases in each hidden/output neuron.

---


