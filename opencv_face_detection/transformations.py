import cv2 as cv
import numpy as np

def rescaleFrame(frame,scale=0.20):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

im = cv.imread('opencv_face_detection/photos/park.jpg')
img = rescaleFrame(im)
cv.imshow('Dog',img)

#translation
def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

translated = translate(img,100,100)
cv.imshow('Translated',translated)

#rotation
def rotate(img,angle,rotPoint=None):
    (height,width) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)
    return cv.warpAffine(img,rotMat,dimensions)

rotated = rotate(img,45)
cv.imshow('Rotated',rotated)
cv.waitKey(0)