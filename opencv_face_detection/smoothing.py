import cv2 as cv

#reading images
img = cv.imread('opencv_face_detection/photos/dog.jpeg')
cv.imshow('Dog',img)

#Averaging
average = cv.blur(img,(3,3))
cv.imshow('Average Blur',average)

#gaussian blur
gauss = cv.GaussianBlur(img,(3,3),0)
cv.imshow('Gaussian Blur',gauss)

#median blur
median = cv.medianBlur(img,3)
cv.imshow('Median Blur',median)

#bilateral blur
#used mostly in advanced computer vision projects
bilateral = cv.bilateralFilter(img,10,35,25)
cv.imshow('Bilateral Blur',bilateral)

cv.waitKey(0)