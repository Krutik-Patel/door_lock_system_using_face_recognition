# DOOR LOCK SYSTEM USING FACE RECOGNITION

## What it does?
This project takes a users face details and detects it using deep learning. Also there is a fun pygame app that takes in your username and detects your face, if the face detected is of username entered, then it unlocks the door.


### Libraries you will need

1) Tensorflow
2) Keras (although it is included with tensorflow 2 onwards)
3) OpenCV
4) pickle
5) pygame
6) numpy
7) pillow

Make sure that you install all the libraries prior to running the program

### Taking user's images for learning

Run the capture.py file, enter your username in the terminal and let the window open. The window will detect a face and a coloured box will appear around your face. It will take 100 pictures of your face in grayscale and save them in png format. All the photos will be stored in the images directory in appropriate username's folder. If you wish to close the window, press 'q' key on the keyboard. If the webcam is not starting, anti-virus might be blocking your cam.


### Training images

Once more than enough people have been captured, you can go on with training the model.
1) Run the train.py python file.
2) It may take some time (the time depends on the number of people who have been captured).
3) Check with the accuracy, if you feel the accuray is less, you may run the file again. But beware, not to train the model too many times, or the model may become overfitting.

Sample images is already included in the images folder. If you dont want them, delete all the folders. But make sure that the train.py is run on more than 2 users, else you might face some issues while detecting your face, talked in the following section.

### Checking the detection

Once you are satisfied with the training, you can move on with trial on the detect.py app.

1) Run the detect.py file
2) A window similar to capture.py will appear.
3) Try detecting, your face, your username will be displayed above the blue box


### DOOR LOCK SYSTEM

If your face is detected, properly, you can start with the door.py app.

1) When you run the file, a pygame window will appear. Do as it says. That is, you will have to input your username (IMP: your username must be similar to the one used in images section. If the images folder name is "faulty_nut" put faulty_nut as username), else you might face problems while unlocking the door.
2) After you successfully enter your name, press Space to open the detection app. When the app detects, it finds the name of that person. If that is the username entered, then Viola! THE DOOR UNLOCKS.

To exit the app, press the close button. 
To exit the detection app, press the 'q' key in the keyboard.


### Caution

The data set is too small, so the model may show False Positives.