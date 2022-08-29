import pygame
import cv2 as cv
import tensorflow as tf
import os
import pickle
import numpy as np

class Door(pygame.sprite.Sprite):
    def __init__(self, image):
        self.sheet = image
        self.animating = False
    def get_image(self, frame, width, height, colour, scale):
        image = pygame.Surface((width, height)).convert_alpha()
        image.blit(self.sheet, (0, 0), (frame * width + 160, 160, width, height))
        image = pygame.transform.scale(image, (width * scale, height * scale))
        image.set_colorkey(colour)
        return image
            
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAS_DIR = os.path.join(BASE_DIR, 'utils', 'data/')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
pygame.init()
clock = pygame.time.Clock()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

cnn = tf.keras.models.load_model(MODEL_DIR)
labels = {}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
cascade = cv.CascadeClassifier(CAS_DIR+'haarcascade_frontalface_alt2.xml')
train_image_dimensions = (152, 152)
cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
pygame.display.set_caption("door")
sprite_door_image = pygame.image.load('./door_images/door_sprite.jpg')
door_sprite = Door(sprite_door_image)
icon = pygame.image.load('./door_images/opened-door-aperture.png')
pygame.display.set_icon(icon)
BACKGROUND = (255, 255, 255)

door_animation_number = 5
door_animation_list = []
for i in range(door_animation_number):
    door_animation_list.append(door_sprite.get_image(i, 1420, 2308, (255, 255, 255), 0.2))

frame = 0
run = True
open_video_capture = False
while run:
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                open_video_capture = True
            # door_sprite.animating = True
    if open_video_capture:
        ret, cap_frame = cap.read()
        gray = cv.cvtColor(cap_frame, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.5)
        for x_fr, y_fr, w_fr, h_fr in faces:
            roi_gray = gray[y_fr: y_fr + h_fr, x_fr: x_fr + w_fr]
            roi_gray = cv.resize(roi_gray, train_image_dimensions, interpolation=cv.INTER_LINEAR)
            square_colour = (255, 0, 0)
            stroke = 2
            cv.rectangle(cap_frame, (x_fr, y_fr), (x_fr + w_fr, y_fr + h_fr), square_colour, stroke)
            roi_gray = roi_gray.reshape(-1, 152, 152, 1) / 255
            y_pred = cnn.predict(roi_gray)
            print(y_pred)
            if np.amax(y_pred) > 0.5:
                pred_name = labels[np.argmax(y_pred)]
            else:
                pred_name = 'unrecognized'
            print(pred_name)
        cv.imshow('frame', cap_frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            open_video_capture = False
            

    if (int(frame) >= door_animation_number):
        frame = door_animation_number - 1
        door_sprite.animating = False
    screen.blit(door_animation_list[int(frame)], (SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2 - 100))
    pygame.display.update()
    if door_sprite.animating:
        frame += 0.1
    clock.tick(60)

cap.release()
cv.destroyAllWindows()
quit()