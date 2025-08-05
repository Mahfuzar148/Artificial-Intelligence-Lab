# 🧠 Fully Connected Neural Network (FCNN) 

---

## 📌 1. What is a Fully Connected Neural Network (FCNN)?

A **Fully Connected Neural Network** (also called a **Dense Neural Network**) is a type of artificial neural network where:

- Every neuron in one layer is connected to **every neuron in the next layer**.
- Each connection has a **weight** that determines the strength of the signal.
- Each neuron has a **bias** that shifts the activation output.
- An **activation function** is applied to introduce non-linearity so the network can learn complex patterns.

💡 **Key idea:** FCNN is the simplest and most common neural network structure, and it’s the building block for many advanced architectures.

---

## 📊 2. Diagram of a Simple Neural Network

![Simple Neural Network Diagram](https://github.com/Mahfuzar148/Artificial-Intelligence-Lab/blob/main/Assignment1/simple%20neural%20network.png)


---

🔍 **3. How Does It Work? (Step-by-Step)**

1. **Input Layer**
   The input layer takes in the feature values from your dataset.
   Example: (x₁, x₂) for two input features.

2. **Weighted Sum**
   Each neuron in the next layer computes a weighted sum of its inputs, plus a bias term:
   
   z = (w₁ × x₁) + (w₂ × x₂) + b
   where:

   * w = weight
   * b = bias

4. **Activation Function**
   A non-linear function is applied to the weighted sum to introduce non-linearity, allowing the network to learn complex patterns. Common activation functions include:

   * ReLU: f(z) = max(0, z)
   * Sigmoid: f(z) = 1 / (1 + e^(-z))
   * Softmax: Used for multi-class probability distribution.

5. **Hidden Layers**
   These are multiple layers of neurons between the input and output layers. Each hidden layer processes and transforms the data further, enabling the network to capture higher-level and more abstract features.

6. **Output Layer**
   The final layer produces the model’s prediction:

   * For classification: Outputs probability scores.
   * For regression: Outputs a continuous numeric value.

7. **Training**
   The learning process uses:

   * Backpropagation to calculate the gradients for each weight and bias.
   * Gradient Descent (or a variant) to update the weights in the direction that minimizes the loss function.

---


## 🎯 4. When to Use FCNN?

✅ **Good for:**
- Tabular data (structured datasets)
- Simple pattern recognition
- Problems where features are not sequential or spatial
- Small to medium-sized datasets

⚠️ **Not ideal for:**
- Image data → Convolutional Neural Networks (CNN) perform better
- Sequential data → Recurrent Neural Networks (RNN), Transformers are better
- Very large datasets → May overfit without regularization

---

## 🚀 5. Why FCNN Can Perform Well

- Learns **complex relationships** between inputs and outputs.
- Flexible — works with many types of problems.
- Easy to implement and train for smaller datasets.
- Fully connected structure ensures **all features are considered**.

---

## 💻 6. Python Implementation with Keras

We’ll build the FCNN as in the diagram:
- **Input layer:** 2 neurons
- **Hidden layer 1:** 2 neurons (ReLU)
- **Hidden layer 2:** 3 neurons (ReLU)
- **Hidden layer 3:** 2 neurons (ReLU)
- **Output layer:** 1 neuron (Softmax in this example)

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


## 📄 7. Model Summary Output

![Model Summary Output](https://github.com/Mahfuzar148/Artificial-Intelligence-Lab/blob/main/Assignment1/simple%20neural%20network%20output.png)

---

## 🧮 8. Code Explanation

### **Layer 1 — Input**

* Shape `(2,)` → accepts 2 features $x_1, x_2$.
* No parameters yet (just placeholders).

---

### **Layer 2 — Hidden Layer 1**

```python
h1 = Dense(2, activation='relu')(inputs)
```

* **2 neurons**, fully connected to input layer.
* **Parameters:** $(2 \times 2) + 2 = 6$.

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

## 🔗 9. Mapping to the Diagram

The structure:

$$
x_1, x_2 \rightarrow h_1(2) \rightarrow h_2(3) \rightarrow h_3(2) \rightarrow y
$$

* **w₁–w₁₈** = weights between neurons.
* **b₁–b₇** = biases in each hidden/output neuron.
* Fully connected means **every neuron** in one layer connects to **every neuron** in the next layer.

---

## ⚠️ 10. Note About Softmax

If you use `softmax` with only **1 neuron**, the output will always be 1.
For binary classification, replace:

```python
outputs = Dense(1, activation='softmax')(h3)
```

with:

```python
outputs = Dense(1, activation='sigmoid')(h3)
```

This outputs probabilities between 0 and 1.

---


```
