'''
Design a Customize Convolutional Neural Network (CNN) for Handwritten Digit Classification with the following specifications:
a) Generate a CNN model with:
i. Two CNN hidden layers (Conv2D) of sizes 32, 64 followed by ReLU Activation.
ii. MaxPooling2D with Kernel size (3, 3), and Stride (1, 1).
iii. Flatten Layer to convert the feature map into 1D with a Dense layer of size 64 followed by an output Dense layer of size 10 with Softmax Activation Function.
b) Display the generated CNN with the required number of parameters.
c) Use the MNIST dataset for training and testing.
d) Adopt Data augmentation (rotation, shift) with the MNIST dataset.
e) Train two CNNs using the original MNIST dataset and augmented MNIST dataset.
f) Use the MNIST dataset as well as the augmented test MNIST dataset to predict the accuracy of the two trained CNNs.
g) Compare and plot the prediction accuracies of the two CNNs.

'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------- Build CNN Model ----------------
def build_model():

    inputs = Input(shape=(28,28,1))

    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = Conv2D(64, (3,3), activation='relu')(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(1,1))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def main():

    # ---------------- Load MNIST ----------------
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # ---------------- Data Augmentation ----------------
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    test_generator_aug = test_datagen.flow(x_test, y_test, batch_size=32, shuffle=False)

    # =====================================================
    # 1️⃣ Train CNN with Original Dataset
    # =====================================================
    print("\nTraining CNN with Original MNIST\n")

    model_original = build_model()
    model_original.summary()

    model_original.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_original.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    loss_orig, acc_orig_test = model_original.evaluate(x_test, y_test)
    loss_orig_aug, acc_orig_augtest = model_original.evaluate(test_generator_aug)

    # =====================================================
    # 2️⃣ Train CNN with Augmented Dataset
    # =====================================================
    print("\nTraining CNN with Augmented MNIST\n")

    model_aug = build_model()
    model_aug.summary()

    model_aug.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_aug.fit(
        train_generator,
        epochs=10,
        verbose=1
    )

    loss_aug, acc_aug_test = model_aug.evaluate(x_test, y_test)
    loss_aug_augtest, acc_aug_augtest = model_aug.evaluate(test_generator_aug)

    # ---------------- Print Results ----------------
    print("\n===== ACCURACY RESULTS =====")
    print("Original Model → Original Test:", acc_orig_test)
    print("Original Model → Augmented Test:", acc_orig_augtest)
    print("Augmented Model → Original Test:", acc_aug_test)
    print("Augmented Model → Augmented Test:", acc_aug_augtest)

    # ---------------- Compare & Plot ----------------
    labels = [
        "Orig→Orig",
        "Orig→AugTest",
        "Aug→Orig",
        "Aug→AugTest"
    ]

    accuracies = [
        acc_orig_test,
        acc_orig_augtest,
        acc_aug_test,
        acc_aug_augtest
    ]

    plt.figure(figsize=(8,5))
    plt.bar(labels, accuracies)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of Two CNNs")
    plt.ylim(0,1)
    plt.show()


if __name__ == "__main__":
    main()