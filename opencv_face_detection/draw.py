import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype='uint8')
cv.imshow('Blank',blank)

# Paint the image a color
blank[200:300,300:400] = 0,0,255
cv.imshow('Red',blank)

# write text on an image
cv.putText(blank,'Yo, Bro',(255,255),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
cv.imshow('Text',blank)

cv.waitKey(0)