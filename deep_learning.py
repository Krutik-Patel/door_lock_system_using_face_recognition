import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


def model_train(x, y, cnn, output):
    x_scaled = x.reshape(-1, 152, 152, 1) / 255
    if (x_scaled.shape[0] == y.shape[0]):
        x_prac, x_test, y_prac, y_test = train_test_split(
            x_scaled, y, test_size=0.2, random_state=42)
        if not cnn:
            print("\n" + "=" * 20)
            print("NO SAVED MODEL FOUND")
            print("=" * 20 + "\n")
            cnn = keras.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(
                    3, 3), input_shape=(152, 152, 1), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(
                    filters=16, kernel_size=(3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(10, activation='relu'),
                keras.layers.Dense(output, activation='softmax')
            ])
            cnn.compile(
                optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy']
            )
        print("\nTraining Dataset:")
        cnn.fit(x_prac, y_prac, epochs=7)
        print("\nTesting Dataset:")
        cnn.evaluate(x_test, y_test)
        return cnn
    else:
        raise TypeError('x and y are not of the same length')


def model_predict(img_array, cnn):
    img_array_scaled = img_array.reshape(-1, 152, 152, 1) / 255
    return cnn.predict(img_array_scaled)
