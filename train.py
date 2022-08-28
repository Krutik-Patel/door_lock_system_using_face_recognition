import os
import numpy as np
import tensorflow as tf
from deep_learning import model_train
import cv2 as cv
from PIL import Image

currID = 0
labels = {}
y = np.array([]) 
x = np.array([]) 

train_image_dimensions = (152, 152)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'images')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')

for root, dirs, files in os.walk(IMG_DIR):
    for filer in files:
        if filer.endswith('png') or filer.endswith('jpg'):
            path = os.path.join(root, filer)
            label = os.path.basename(root).replace(' ', '_').lower()
            if label not in labels.values():
                labels[currID] = label
                currID += 1
            image = cv.imread(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
            resized_img = cv.resize(image, train_image_dimensions, interpolation=cv.INTER_LINEAR)
            x = np.append(x, resized_img)
            y = np.append(y, currID - 1)


if os.path.exists(MODEL_DIR):
    model = tf.keras.models.load_model(MODEL_DIR)
else:
    model = None
model = model_train(x, y, model)
model.save(os.path.join(MODEL_DIR))
quit()