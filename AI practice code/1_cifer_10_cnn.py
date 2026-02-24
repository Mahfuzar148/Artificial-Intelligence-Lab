import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =====================================================
# 1️⃣ Load CIFAR-10 from local folder
# =====================================================
def load_batch(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    data = dict[b'data']
    labels = dict[b'labels']
    return data, labels

path = "cifar-10-batches-py"

# ---- Load Training Batches (1–5) ----
X = []
y = []

for i in range(1,6):
    data, labels = load_batch(os.path.join(path, f"data_batch_{i}"))
    X.append(data)
    y += labels

X = np.concatenate(X)
y = np.array(y)

# ---- Load Test Batch ----
X_test, y_test = load_batch(os.path.join(path, "test_batch"))
X_test = np.array(X_test)
y_test = np.array(y_test)

# =====================================================
# 2️⃣ Reshape (3072 → 32×32×3)
# =====================================================
X = X.reshape(-1,3,32,32).transpose(0,2,3,1)
X_test = X_test.reshape(-1,3,32,32).transpose(0,2,3,1)

# =====================================================
# 3️⃣ Preprocessing (Normalization)
# =====================================================
X = X / 255.0
X_test = X_test / 255.0

# =====================================================
# 4️⃣ Train / Validation Split
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 5️⃣ Data Augmentation (Only for training)
# =====================================================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True
)

# =====================================================
# 6️⃣ CNN Model
# =====================================================
inputs = Input(shape=(32,32,3))

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

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# 7️⃣ Early Stopping
# =====================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =====================================================
# 8️⃣ Train Model
# =====================================================
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# =====================================================
# 9️⃣ Plot Accuracy & Loss
# =====================================================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")

plt.show()

# =====================================================
# 🔟 Test Set Evaluation
# =====================================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# =====================================================
# 1️⃣1️⃣ Show 10 Test Predictions
# =====================================================
class_names = ['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck']

pred = model.predict(X_test[:10])
pred_classes = np.argmax(pred, axis=1)

plt.figure(figsize=(15,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i])
    plt.title(f"P:{class_names[pred_classes[i]]}\nT:{class_names[y_test[i]]}")
    plt.axis('off')

plt.show()
