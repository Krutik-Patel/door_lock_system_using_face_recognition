import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

def model_train(x, y):
    if (x.shape[0] == y.shape[0]):
        pass
    else:
        x_scaled = x.rescale(-1, 152 , 152, 1) / 255
        x_prac, x_test, y_prac, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state = 42)
        cnn = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(152, 152, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(3, activation='sigmoid')
        ])
        cnn.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )
        cnn.fit(x_prac, y_prac, epochs=7)
        cnn.evaluate(x_test, y_test)
        

