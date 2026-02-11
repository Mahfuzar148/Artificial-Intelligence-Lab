# Topic: How to Count Parameters in Neural Networks (Dense Layers)

---

## 1. Introduction

In neural networks, **parameters** are the values that the model learns during training. These include:

* Weights
* Biases

Understanding how to count parameters in each layer is very important for:

* Model analysis
* Exam preparation
* Interview questions
* Understanding model complexity

This document explains how to calculate the number of parameters in Dense (Fully Connected) layers step by step.

---

## 2. What Are Parameters?

In a Dense layer, every input neuron is connected to every output neuron.

Each connection has:

* One weight
* Each output neuron also has one bias

So total parameters = total weights + total biases

---

## 3. Formula for Dense Layer Parameters

For a Dense layer:

Parameters = (Input Units × Output Units) + Output Units

Or simply:

Parameters = (Weights) + (Biases)

Where:

* Weights = input_neurons × output_neurons
* Biases = number of output_neurons

---

## 4. Why This Formula Works

Consider a Dense layer:

Input size = 3
Output size = 4

Each of the 3 input neurons connects to all 4 output neurons.

So total weight connections:

3 × 4 = 12 weights

Each output neuron has 1 bias:

4 biases

Total parameters:

12 + 4 = 16

---

## 5. Example Model Parameter Counting

Consider this model:

Input((3,))
Dense(4)
Dense(8)
Dense(4)
Dense(1)

We calculate layer by layer.

---

### Layer 1: Dense(4)

Input neurons = 3
Output neurons = 4

Parameters = (3 × 4) + 4
= 12 + 4
= 16

---

### Layer 2: Dense(8)

Input neurons = 4
Output neurons = 8

Parameters = (4 × 8) + 8
= 32 + 8
= 40

---

### Layer 3: Dense(4)

Input neurons = 8
Output neurons = 4

Parameters = (8 × 4) + 4
= 32 + 4
= 36

---

### Output Layer: Dense(1)

Input neurons = 4
Output neurons = 1

Parameters = (4 × 1) + 1
= 4 + 1
= 5

---

## 6. Total Parameters

Total = 16 + 40 + 36 + 5
Total = 97

So the model will show:

Total params: 97
Trainable params: 97
Non-trainable params: 0

---

## 7. Understanding Weight Matrix Shape

For each Dense layer:

Weight matrix shape = (input_units, output_units)
Bias vector shape = (output_units,)

Example:

Dense(4) with input 3 →

Weight matrix shape = (3, 4)
Bias shape = (4,)

---

## 8. Quick Shortcut for Exams

For any Dense layer:

Input × Output + Output

Remember:

Multiply, then add output neurons.

---

## 9. Important Notes

1. Input layer has no parameters.
2. Only layers with learnable weights (Dense, Conv, etc.) have parameters.
3. More parameters = more model capacity.
4. Too many parameters can cause overfitting.

---

## 10. Conclusion

Parameter counting helps understand:

* Model size
* Learning complexity
* Computational cost

For Dense layers, the formula is simple:

Parameters = (Input Units × Output Units) + Output Units

This includes both weights and biases.

---

End of Document
