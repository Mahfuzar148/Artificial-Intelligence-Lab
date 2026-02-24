from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(28,28),
    color_mode='grayscale',   # RGB হলে remove করো
    batch_size=32,
    class_mode='sparse'
)

test_data = train_datagen.flow_from_directory(
    'dataset/test',
    target_size=(28,28),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse'
)