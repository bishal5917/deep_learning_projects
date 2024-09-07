import cv2 as cv
import numpy as np

#reading images
img = cv.imread('opencv_face_detection/photos/dog.jpeg')
cv.imshow('Dog',img)

blank = np.zeros(img.shape[:2],dtype='uint8')
cv.imshow('Blank Image',blank)

mask = cv.circle(blank.copy(),(img.shape[1]//2+45,img.shape[0]//2),100,255,-1)
cv.imshow('Mask',mask)

masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('Masked Image',masked)

cv.waitKey(0)