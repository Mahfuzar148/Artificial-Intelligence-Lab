import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    x = Dense(512,activation='relu')(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    
    outputs = Dense(10,activation='softmax')(x)
    
    model = Model(inputs,outputs)
    return model


def main():
    
    # ----------------- Data Generator -----------------
    train_gen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.1
    )

    test_gen = ImageDataGenerator(rescale=1/255.0)

    # ----------------- Load Data -----------------
    train_data = train_gen.flow_from_directory(
        'mnist-png/train',
        batch_size=32,
        target_size=(28,28),
        class_mode='sparse',
        color_mode='grayscale',
        subset='training'
    )

    val_data = train_gen.flow_from_directory(
        'mnist-png/train',
        batch_size=32,
        target_size=(28,28),
        class_mode='sparse',
        color_mode='grayscale',
        subset='validation'
    )

    test_data = test_gen.flow_from_directory(
        'mnist-png/test',
        batch_size=32,
        target_size=(28,28),
        class_mode='sparse',
        color_mode='grayscale',
        shuffle=False
    )

    # ----------------- Build Model -----------------
    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ----------------- Train -----------------
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[early_stop],
        verbose=1
    )

    # ----------------- Evaluate -----------------
    loss, acc = model.evaluate(test_data)
    print('Accuracy : ', acc)
    print('Loss : ', loss)

    # ----------------- Show 10 Odd Digit Predictions -----------------
    odd_images = []
    odd_labels = []

    test_data.reset()

    while len(odd_images) < 10:
        images, labels = next(test_data)

        for i in range(len(labels)):
            if int(labels[i]) % 2 == 1:   # odd condition
                odd_labels.append(int(labels[i]))
                odd_images.append(images[i])

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