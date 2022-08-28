import cv2 as cv
import os
from deep_learning import model_predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAS_DIR = os.path.join(BASE_DIR, 'utils', 'data/')
cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
cascade = cv.CascadeClassifier(CAS_DIR+'haarcascade_frontalface_alt2.xml')
labels = {}

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.5)
        for x, y, w, h in faces:
            roi_gray = gray[y: y + h, x: x + w]
            colour = (0, 0, 255)
            stroke = 2
            cv.rectangle(frame, (x, y), (x + w, y + h), colour, stroke)
            y = model_predict(roi_gray)
            pred_name = labels[np.argmax(y)]
            colour = (255, 255, 255)
            font = cv.FONT_HERSHEY_COMPLEX
            cv.putText(frame, pred_name, (x, y), font, 1, colour, stroke, cv.LINE_AA)
        cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()