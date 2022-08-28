import cv2 as cv
import os
from deep_learning import model_predict
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAS_DIR = os.path.join(BASE_DIR, 'utils', 'data/')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
cascade = cv.CascadeClassifier(CAS_DIR+'haarcascade_frontalface_alt2.xml')
labels = {}
train_image_dimensions = (152, 152)
cnn = tf.keras.models.load_model(MODEL_DIR)

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.5)
        for x, y, w, h in faces:
            roi_gray = gray[y: y + h, x: x + w]
            roi_gray = cv.resize(roi_gray, train_image_dimensions, interpolation=cv.INTER_LINEAR)
            colour = (0, 0, 255)
            stroke = 2
            cv.rectangle(frame, (x, y), (x + w, y + h), colour, stroke)
            roi_gray = roi_gray.reshape(-1, 152, 152, 1) / 255
            y = cnn.predict(roi_gray)
            pred_name = np.argmax(y)
            print(pred_name)
            colour = (255, 255, 255)
            font = cv.FONT_HERSHEY_COMPLEX
            # cv.putText(frame, pred_name, (x, y), font, 1, colour, stroke, cv.LINE_AA)
        cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()