import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model():
    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def main():

    # ✅ Load MNIST dataset
    (X, y), (x_test, test_labels) = tf.keras.datasets.mnist.load_data()

    print("Original shape:", X.shape)

    # ✅ Reshape for CNN (add channel dimension)
    X = X.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # ✅ Normalize
    X = X / 255.0
    x_test = x_test / 255.0

    # ✅ Train/Validation split
    x_train, x_val, train_labels, val_labels = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42
    )

    # ✅ Data Augmentation (optional but helpful)
    dataGen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        dataGen.flow(x_train, train_labels, batch_size=64),
        epochs=10,
        validation_data=(x_val, val_labels),
        callbacks=[early_stop]
    )

    # ✅ Evaluation
    loss, acc = model.evaluate(x_test, test_labels)

    print("Test Accuracy:", acc)
    print("Test Loss:", loss)

    # ✅ Prediction
    pred = model.predict(x_test[:10])
    predict_class = np.argmax(pred, axis=1)
    true_class = test_labels[:10]

    # ✅ Plot
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 2, 1)
    plt.title('Accuracy curve')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.title('Loss curve')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()

    for i in range(2, 10):
        plt.subplot(4, 2, i+1)
        plt.title(f'True: {true_class[i-2]}  Pred: {predict_class[i-2]}')
        plt.imshow(x_test[i-2].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()