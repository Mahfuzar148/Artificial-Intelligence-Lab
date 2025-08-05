from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Input layer with 2 features
inputs = Input((2,))

# Hidden layers
h1 = Dense(2, activation='relu')(inputs)
h2 = Dense(3, activation='relu')(h1)
h3 = Dense(2, activation='relu')(h2)

# Output layer (sigmoid for binary classification)
outputs = Dense(1, activation='sigmoid')(h3)

# Build and summarize the model
model = Model(inputs, outputs)
model.summary()
