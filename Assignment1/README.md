---

# 📚 Fully Connected Neural Network (FCNN) 

---

## **1. Introduction**

A **Fully Connected Neural Network** (also called a **Dense Neural Network**) is a type of Artificial Neural Network where **each neuron in one layer is connected to every neuron in the next layer**.

* Each connection has:

  * **Weight (w)**: Determines the strength and direction of the connection.
  * **Bias (b)**: Allows shifting of the activation function.
* An **activation function** is applied to the neuron’s output to introduce non-linearity.

💡 **Key idea:** This structure ensures that every feature from one layer is considered in the next layer’s calculations.

---

## **2. Architecture**

A typical FCNN has:

1. **Input Layer**

   * Receives the data features.
   * Each node represents one input feature.

2. **Hidden Layers**

   * One or more fully connected layers.
   * Each neuron receives inputs from all neurons in the previous layer.
   * Each layer applies a **weighted sum + bias**, then passes it through an **activation function**.

3. **Output Layer**

   * Produces the final prediction (classification or regression).
   * Uses an appropriate activation function:

     * **Softmax** → Multi-class classification
     * **Sigmoid** → Binary classification
     * **Linear** → Regression


---

### **3. How It Works — Step-by-Step**

---

#### **1. Forward Pass**

For each neuron in a layer, the **weighted sum** is calculated as:

$$
z = \sum_{i=1}^{n} w_i \cdot x_i + b
$$

Then, the **activation function** is applied:

$$
a = f(z)
$$

Where:

* $f$ can be **ReLU**, **Sigmoid**, **Softmax**, or another activation function depending on the task.

---

#### **2. Loss Calculation**

The predicted output is compared with the actual target value using a **loss function**, such as:

* **Mean Squared Error (MSE)** → For regression tasks
* **Cross-Entropy Loss** → For classification tasks

---

#### **3. Backpropagation**

* Compute the **gradients** of the loss with respect to each weight and bias.
* Use the **chain rule** to propagate the error backward through the network, layer by layer.

---

#### **4. Weight Update**

The weights and biases are updated using an optimization algorithm (e.g., **Gradient Descent**):

$$
w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}
$$

Where:

* $\eta$ is the **learning rate** — it controls how big each update step is.
* $\frac{\partial L}{\partial w}$ is the **gradient of the loss** with respect to the weight.

---

---

## **4. Activation Functions in FCNN**

* **ReLU**: $f(z) = \max(0, z)$ → prevents vanishing gradients, widely used in hidden layers.
* **Sigmoid**: Outputs values between 0 and 1 → useful for binary outputs.
* **Softmax**: Converts outputs into probabilities that sum to 1 → used for multi-class classification.

---

## **5. Advantages**

✅ Simple and easy to implement.
✅ Can model complex relationships if enough layers and neurons are used.
✅ Works well for structured/tabular data.
✅ Serves as a foundation for more complex networks (CNNs, RNNs).

---

## **6. Disadvantages**

⚠️ Large number of parameters → can lead to overfitting.
⚠️ Inefficient for image or sequential data (better to use CNNs or RNNs).
⚠️ Computationally expensive for very large input sizes.

---

## **7. When to Use**

* Tabular datasets (structured data).
* Problems where all features are important.
* Small to medium-sized datasets.
* As the classification/regression head in more complex architectures.

---

## **8. Python Example with Keras**

```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Input layer with 2 features
inputs = Input((2,))

# Hidden layers
h1 = Dense(2, activation='relu')(inputs)
h2 = Dense(3, activation='relu')(h1)
h3 = Dense(2, activation='relu')(h2)

# Output layer (sigmoid for binary classification)
outputs = Dense(1, activation='sigmoid')(h3)

# Build and summarize the model
model = Model(inputs, outputs)
model.summary()
```

---

## **9. Parameter Calculation Example**

| Layer     | Inputs | Neurons | Parameters Formula | Parameters |
| --------- | ------ | ------- | ------------------ | ---------- |
| H1        | 2      | 2       | `(2 × 2) + 2`      | 6          |
| H2        | 2      | 3       | `(2 × 3) + 3`      | 9          |
| H3        | 3      | 2       | `(3 × 2) + 2`      | 8          |
| Out       | 2      | 1       | `(2 × 1) + 1`      | 3          |
| **Total** | —      | —       | —                  | **26**     |

---

## **10. Visual Representation**

**Forward Pass Diagram:**

1. Input layer → Hidden layers (fully connected) → Output layer.
2. Every neuron in one layer connects to **every neuron** in the next layer.

---


