#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:15:13 2020

@author: sharad
"""

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2

newmod=load_model('hand_gestures.h5')
# This background will be a global variable that we update through a few functions
background = None

# Start with a halfway point between 0 and 1 of accumulated weight
accumulated_weight = 0.5

# Manually set up our ROI for grabbing the hand.
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

#For removing the background from the foreground
def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed the weighted average of the image passed in.
    '''
    # Grab the background
    global background
    
    # For first time, create the background from a copy of the frame.
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)

#For returning the hand segment
def segment(frame, threshold=20):
    global background
    
    # Calculates the Absolute Differentce between the backgroud and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the image so we can grab the foreground
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours form the image
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list is 0, then we didn't grab any contours!
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)
 
#For predicting the Image 
def thres_display(img):
    width=64
    height=64
    dim=(width,height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img=image.img_to_array(resized)
    test_img=np.expand_dims(test_img,axis=0)
    result= newmod.predict(test_img)
    val=[index for index,value in enumerate(result[0]) if value ==1]
    return val
    
cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0

# keep looping, until interrupted
while True:
    # get the current frame
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame

    # Grab the ROI from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 30 frames we will calculate the average of the background.
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
    else:
        # now that we have the background, we can segment the hand.
        cv2.putText(frame_copy, "Place your hand in side the box", (330, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 0: Fist", (330, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 1: Five", (330, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 2: None", (330, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 3: Okay", (330, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 4: Peace", (330, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 5: Rad", (330, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 6: Straight", (330, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "Index 7: Thumbs", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)

        # segment the hand region
        hand = segment(gray)

        # First check if we were able to actually detect a hand
        if hand is not None:
            thresholded, hand_segment = hand
            
            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # display the thresholded image
            cv2.imshow("Thresholded Image", thresholded)
            res=thres_display(thresholded)
            
            if len(res)==0:
                cv2.putText(frame_copy, str('None'), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                x='index'+str(res[0])
                cv2.putText(frame_copy, str(x), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 2)

    # increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Hand Gestures", frame_copy)

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()