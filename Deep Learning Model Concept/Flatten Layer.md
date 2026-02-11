# Topic: Flatten Layer in Deep Learning (Keras / TensorFlow)

---

## 1. Introduction

The **Flatten layer** is a simple but very important layer in deep learning models, especially when working with image data.

It is commonly used when transitioning from:

* Convolutional layers (Conv2D)
* Multi-dimensional image input

To:

* Fully Connected (Dense) layers

The Flatten layer does not learn anything. It only reshapes the input data.

---

## 2. Why Flatten Layer Is Needed

Dense layers require 2D input in the form:

(batch_size, features)

But image data is usually 3D per sample:

(height, width, channels)

When batch is included, the full shape becomes:

(batch_size, height, width, channels)

Since Dense layers cannot directly process 4D input, we use Flatten to convert multi-dimensional data into a single long vector.

---

## 3. What Flatten Does

Flatten reshapes input without changing data values.

It converts:

(height × width × channels)

Into:

(height × width × channels,)

It does NOT:

* Change pixel values
* Add weights
* Add bias
* Perform computation

It only rearranges data.

---

## 4. Simple Example

### Example Input

Suppose we have a small image of shape:

(2, 2, 1)

Matrix:

[[1, 2],
[3, 4]]

After Flatten:

[1, 2, 3, 4]

Shape changes from:

(2, 2, 1)

To:

(4,)

---

## 5. MNIST Example

Input shape:

(28, 28, 1)

Total pixels:

28 × 28 × 1 = 784

After Flatten:

(784,)

With batch:

Before Flatten:
(None, 28, 28, 1)

After Flatten:
(None, 784)

---

## 6. Code Example

```python
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

inputs = Input((28, 28, 1))
x = Flatten()(inputs)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
```

---

## 7. Does Flatten Have Parameters?

No.

Flatten layer:

* Has 0 parameters
* Is not trainable
* Only reshapes data

In model summary you will see:

Flatten (Flatten)      (None, 784)      0

---

## 8. When to Use Flatten

Use Flatten when:

* Moving from convolutional layers to dense layers
* Converting image data into feature vector
* Building simple DNN for image classification

---

## 9. When Flatten Is Not Ideal

Flatten removes spatial structure of images.

This means:

* Neighbor pixel relationships are lost
* Spatial patterns are not preserved

In modern CNN architectures, instead of Flatten, we often use:

* GlobalAveragePooling2D
* GlobalMaxPooling2D

These reduce dimensions while preserving important information.

---

## 10. Advantages

* Simple
* No computation cost
* Easy integration with Dense layers

---

## 11. Disadvantages

* Destroys spatial information
* Can increase number of parameters significantly
* May cause overfitting in large images

---

## 12. Summary

Flatten layer:

* Converts multi-dimensional input into 1D vector
* Has no trainable parameters
* Is used before Dense layers
* Acts as a bridge between convolutional and fully connected layers

---

### Viva Ready Definition

Flatten layer reshapes multi-dimensional input data into a one-dimensional vector so that it can be processed by Dense layers. It has no trainable parameters and does not modify the data values.

---

End of Document
