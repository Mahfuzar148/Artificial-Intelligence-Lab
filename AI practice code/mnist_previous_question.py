import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():

    # ---------------- Load MNIST ----------------
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # ---------------- Keep Only Odd Digits ----------------
    odd_digits = [1,3,5,7,9]

    train_mask = np.isin(y_train, odd_digits)
    test_mask = np.isin(y_test, odd_digits)

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    # ---------------- Normalize ----------------
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # ---------------- 85% Training ----------------
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_train, y_train, test_size=0.15, random_state=42
    )

    # 15% Validation (of training set)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=42
    )

    print("Train:", x_train.shape)
    print("Validation:", x_val.shape)

    # ---------------- Data Augmentation ----------------
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=32
    )

    val_generator = val_datagen.flow(
        x_val, y_val,
        batch_size=32
    )

    test_generator = test_datagen.flow(
        x_test, y_test,
        batch_size=32,
        shuffle=False
    )

    # ---------------- Build Model ----------------
    inputs = Input(shape=(28,28,1))

    x = Conv2D(32,(3,3),activation='relu')(inputs)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = Conv2D(128,(3,3),activation='relu')(x)

    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    outputs = Dense(10,activation='softmax')(x)

    model = Model(inputs,outputs)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ---------------- Callbacks ----------------
    checkpoint = ModelCheckpoint(
        "best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ---------------- First 10 Epochs ----------------
    print("\nTraining Full Network (First 10 Epochs)\n")

    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # ---------------- Freeze First 3 Conv Layers ----------------
    print("\nFreezing First 3 Conv Layers\n")

    model.layers[1].trainable = False
    model.layers[2].trainable = False
    model.layers[3].trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ---------------- Next 20 Epochs ----------------
    model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # ---------------- Load Best Model ----------------
    model.load_weights("best_model.keras")

    # ---------------- Final Evaluation ----------------
    loss, acc = model.evaluate(test_generator)

    print("\nFinal Test Accuracy (Odd Digits Only):", acc)


if __name__ == "__main__":
    main()