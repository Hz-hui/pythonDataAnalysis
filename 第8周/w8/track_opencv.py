import cv2 as cv
import numpy as np
cap = cv.VideoCapture(1)# 注意，不同机器上可能指代的摄像头不一样，可以试0或1
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV 110，130 red 0 20
    lower_blue = np.array([110,50,50])#np.array([156,43,46]) #np.array([110,50,50])
    upper_blue = np.array([130,255,255])#np.array([180,255,255])#np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
