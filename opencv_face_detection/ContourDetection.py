import cv2 as cv

img = cv.imread('opencv_face_detection/photos/dog.jpeg')
cv.imshow('Dog',img)

# converting to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Blur
blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

# Edge cascade
canny = cv.Canny(blur,125,175)
cv.imshow('Canny Edges',canny)

contours,hierarchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found')

cv.waitKey(0)