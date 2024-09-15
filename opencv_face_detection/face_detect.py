import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('opencv_face_detection/photos/man.jpg')
# cv.imshow('Man',img)

#converting to grayscale, coz we don't need colors
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

#using haar_cascade
haar_cascade = cv.CascadeClassifier('opencv_face_detection/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

#drawing rectangle over faces
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Result',img)
#pring the number of faces found
print(f'Number of faces: {len(faces_rect)}')

cv.waitKey(0)