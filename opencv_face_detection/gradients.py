import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Histograms allows us to visualize the intensity of color distribution in images
#reading images
img = cv.imread('opencv_face_detection/photos/dog.jpeg')
cv.imshow('Dog',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#laplacian
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('Laplacian',lap)

cv.waitKey(0)