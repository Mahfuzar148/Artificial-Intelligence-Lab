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


---

## **Code Explanation — Step by Step**

---

### **Import Statements**

```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
```

**1. `tensorflow.keras.layers.Dense`**

* **What it is:** A *Dense Layer* (also called a *Fully Connected Layer*) where **each neuron** is connected to **all neurons** in the previous layer.
* **When to use:**

  * When you want every neuron to see the whole previous layer’s output.
  * Works well for structured/tabular data, simple neural networks, or the final layers in CNNs.
* **Common syntax:**

  ```python
  Dense(units, activation=None, use_bias=True)
  ```

  * `units`: Number of neurons in the layer.
  * `activation`: Activation function name (e.g., `"relu"`, `"sigmoid"`, `"softmax"`).
  * `use_bias`: Whether to add a bias term.

---

**2. `tensorflow.keras.layers.Input`**

* **What it is:** The starting point (placeholder) for your model’s input data.
* **When to use:**

  * Always in the **Functional API** style models.
  * Needed to define the **shape** of the incoming data.
* **Common syntax:**

  ```python
  Input(shape, name=None)
  ```

  * `shape`: Tuple of feature dimensions (excluding batch size).
  * Example: `Input((2,))` means **2 features** as input.

---

**3. `tensorflow.keras.models.Model`**

* **What it is:** A model class in the Keras Functional API that connects **inputs** to **outputs**.
* **When to use:**

  * When building models with complex topologies (e.g., multiple inputs/outputs).
  * More flexible than `Sequential`.
* **Common syntax:**

  ```python
  Model(inputs=..., outputs=..., name=None)
  ```

  * `inputs`: Input layer(s).
  * `outputs`: Output layer(s).

---

### **Defining the Model**

---

#### **Step 1 — Input Layer**

```python
inputs = Input((2,))
```

* **Shape:** `(2,)` → means 2 input features: $x_1, x_2$.
* **What it does in this code:** Sets up the placeholder for your input data so the model knows the shape before training starts.

---

#### **Step 2 — First Hidden Layer**

```python
h1 = Dense(2, activation='relu')(inputs)
```

* **`Dense(2, activation='relu')`**:

  * 2 neurons in this layer.
  * Activation function: **ReLU** (`f(z) = max(0, z)`).
  * Learns **first-level features** from the raw input.
* **`(inputs)`**:

  * Passes the `inputs` layer’s output to this layer.
* **Parameters here:** `(2 inputs × 2 neurons) + 2 biases = 6 parameters`.

---

#### **Step 3 — Second Hidden Layer**

```python
h2 = Dense(3, activation='relu')(h1)
```

* 3 neurons, connected to **all outputs of h1**.
* ReLU activation again for non-linearity.
* **Parameters:** `(2 neurons from h1 × 3) + 3 biases = 9 parameters`.
* **Purpose in code:** Learns more complex combinations of the features from h1.

---

#### **Step 4 — Third Hidden Layer**

```python
h3 = Dense(2, activation='relu')(h2)
```

* 2 neurons.
* Fully connected to h2’s 3 outputs.
* **Parameters:** `(3 × 2) + 2 = 8 parameters`.
* **Purpose in code:** Extracts refined patterns before sending them to the output layer.

---

#### **Step 5 — Output Layer**

```python
outputs = Dense(1, activation='softmax')(h3)
```

* **1 neuron** → final prediction output.
* Activation: **Softmax** (usually for multi-class classification).
  ⚠️ With **1 neuron**, softmax always outputs `1`. For binary classification, you’d use:

  ```python
  outputs = Dense(1, activation='sigmoid')(h3)
  ```
* **Parameters:** `(2 × 1) + 1 bias = 3 parameters`.
* **Purpose in code:** Converts last layer output into the final prediction.

---

### **Building the Model**

```python
model = Model(inputs, outputs)
```

* Links the `inputs` layer to the `outputs` layer.
* Creates a **trainable computational graph** from start to finish.

---

### **Displaying the Model Summary**

```python
model.summary()
```

* Shows:

  * Layer names & output shapes.
  * Number of parameters (weights + biases).
  * Whether layers are trainable.

---

## **Parameter Recap for This Model**

| Layer     | Neurons | Parameters Formula | Parameters |
| --------- | ------- | ------------------ | ---------- |
| H1        | 2       | `(2 × 2) + 2`      | 6          |
| H2        | 3       | `(2 × 3) + 3`      | 9          |
| H3        | 2       | `(3 × 2) + 2`      | 8          |
| Out       | 1       | `(2 × 1) + 1`      | 3          |
| **Total** | —       | —                  | **26**     |

---



```
