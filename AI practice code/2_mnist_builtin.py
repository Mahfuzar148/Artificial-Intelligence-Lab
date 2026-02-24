import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model


def build_model():
    
    inputs = Input(shape=(28,28,1))
    
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    
    x = Dense(256,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    
    outputs = Dense(10,activation='softmax')(x)
    
    model = Model(inputs,outputs)
    return model


def main():

    # ---------------- Load MNIST ----------------
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # ---------------- Normalize ----------------
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # ---------------- Expand Dimension ----------------
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    # ---------------- Build Model ----------------
    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # ---------------- Train ----------------
    history = model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ---------------- Evaluate ----------------
    loss, acc = model.evaluate(x_test, y_test)
    print("Test Accuracy:", acc)
    print("Test Loss:", loss)

    # ---------------- Show 10 Odd Digit ----------------
    odd_images = []
    odd_labels = []

    for i in range(len(y_test)):
        if y_test[i] % 2 == 1:
            odd_images.append(x_test[i])
            odd_labels.append(y_test[i])
        if len(odd_images) == 10:
            break

    prediction = model.predict(np.array(odd_images))
    pred_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)

    plt.figure(figsize=(18,12))

    for i in range(10):
        plt.subplot(5,2,i+1)
        plt.imshow(odd_images[i].reshape(28,28), cmap='gray')
        plt.title(
            f"True: {odd_labels[i]}\n"
            f"Pred: {pred_class[i]}\n"
            f"Conf: {confidence[i]*100:.1f}%"
        )
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()