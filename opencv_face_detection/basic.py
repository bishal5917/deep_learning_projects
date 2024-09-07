import cv2 as cv
import numpy as np

def rescaleFrame(frame,scale=0.20):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

img = cv.imread('opencv_face_detection/photos/park.jpg')
resizedImg = rescaleFrame(img)
cv.imshow('Dog',resizedImg)

# converting to grayscale
gray = cv.cvtColor(resizedImg,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Blur
blur = cv.GaussianBlur(resizedImg,(7,7),cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

# Edge cascade
canny = cv.Canny(resizedImg,125,175)
cv.imshow('Canny Edges',canny)

# Dilated
dilated = cv.dilate(canny,(7,7),iterations=3)
cv.imshow('Dilated',dilated)

#Resizing with opencv function
resized = cv.resize(img,(500,600),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)

#cropping the image
cropped = img[50:200,200:400]
cv.imshow('Cropped',cropped)

cv.waitKey(0)