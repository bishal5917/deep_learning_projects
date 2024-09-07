import cv2 as cv

#reading images
# img = cv.imread('opencv_face_detection/photos/dog.jpeg')
# cv.imshow('Dog',img)
# cv.waitKey(0)

#reading videos
capture = cv.VideoCapture('opencv_face_detection/videos/dog_mirror_video.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()