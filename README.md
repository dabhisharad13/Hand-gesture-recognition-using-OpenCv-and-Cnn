# Hand-gesture-recognition-using-OpenCv-and-Cnn
This project is the combination of OpenCv and Cnn model. OpenCv is use to capture the current frame from your webcam and further Cnn is use to classify the image in the current frame.
The dataset used has 7 hand gestures

0: Fist, 1: Five, 2: None, 3: Okay, 4: Peace, 5: Rad, 6: Straight, 7: Thumbs

# How to run
1. Firstly run the Cnn file to generate the model 
2. Use the generated model in the Hand-gesture-OpenCv.py
3. Run the Hand-gesture-OpenCv file in spyder
4. A window will popup which will calculate the background, wait for the further instructions

# Note:
I have used contours to detect the hand segment. When the program is calcualting the background make sure that the background is plain and does not consist any edges or corners else it will take them into consideration. Once the background is calculated don't move the webcam else the background will get distorted. Make sure you have proper light coming at your hand while doing the gestures since contours requires proper light to detect boundary.

# Data-set
The training set consist of 7999 images belonging to 8 classes and test set consist of 4000 images belonging to 8 classes.

Epoch 1/1
7999/7999 [==============================] - 551s 69ms/step - loss: 0.0401 - accuracy: 0.9869 - val_loss: 2.0493 - val_accuracy: 0.9762

