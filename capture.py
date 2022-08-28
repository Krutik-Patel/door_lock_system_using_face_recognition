import cv2 as cv
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAS_DIR = os.path.join(BASE_DIR, 'utils', 'data/')
cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
cascade = cv.CascadeClassifier(CAS_DIR+'haarcascade_frontalface_alt2.xml')
if not os.path.exists(os.path.join(BASE_DIR, 'images')):
    os.makedirs(os.path.join(BASE_DIR, 'images'))
IMG_DIR = os.path.join(BASE_DIR, 'images')
user = input("Test subject name: ")

count = 0
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.5)
        for x, y, w, h in faces:
            count += 1
            if count > 100:
                break
            if not os.path.exists(os.path.join(IMG_DIR, f'gray_image_{user}')):
                os.makedirs(os.path.join(IMG_DIR, f'gray_image_{user}'))
            PATH = os.path.join(IMG_DIR, f'gray_image_{user}')
            roi_gray = gray[y:y+h, x:x+w]
            cv.imwrite(f'{PATH}/image_{user}{count}.png', roi_gray)
            rect_colour = (255, 0, 0)
            stroke = 2
            cv.rectangle(frame, (x, y), (x + w, y + h), rect_colour, stroke)

        cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()