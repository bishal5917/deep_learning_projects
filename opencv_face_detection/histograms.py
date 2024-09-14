import cv2 as cv
import matplotlib.pyplot as plt

#Histograms allows us to visualize the intensity of color distribution in images
#reading images
img = cv.imread('opencv_face_detection/photos/dog.jpeg')
cv.imshow('Dog',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Grayscale histogram
gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])

plt.figure()
plt.title('Grayscale histogram')
plt.xlabel('Bins')
plt.ylabel('No of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

cv.waitKey(0)