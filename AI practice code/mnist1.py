import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model():
    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def main():

    # 🔹 Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # 🔹 Load Dataset
    train_data = train_datagen.flow_from_directory(
        'mnist-png/train',
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        class_mode='sparse'
    )

    test_data = test_datagen.flow_from_directory(
        'mnist-png/test',
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )

    # 🔹 Build Model
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

    # 🔹 Train
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=test_data,
        callbacks=[early_stop]
    )

    # 🔹 Evaluate
    loss, acc = model.evaluate(test_data)
    print("Test Accuracy:", acc)
    print("Test Loss:", loss)

    # 🔹 Plot Accuracy & Loss Curve
    plt.figure(figsize=(14, 5))

    # Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 🔹 Show Predictions
    images, labels = next(test_data)
    preds = model.predict(images)
    predicted_classes = np.argmax(preds, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"T:{int(labels[i])}  P:{predicted_classes[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()