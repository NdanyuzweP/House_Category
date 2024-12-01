import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

def build_model(input_shape):
    # Build a Sequential model
    model = models.Sequential()

    # Add input layer with input shape
    model.add(layers.InputLayer(input_shape=input_shape))

    # Add hidden layers with L1 regularization
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01)))

    # Output layer (binary classification for basic vs luxury)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(X_train, y_train, model):
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    return model, history
