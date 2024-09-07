import cv2 as cv

#reading videos
capture = cv.VideoCapture('opencv_face_detection/videos/dog_mirror_video.mp4')

def rescaleFrame(frame,scale=0.20):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

while True:
    isTrue, frame = capture.read()
    resizedFrame = rescaleFrame(frame)
    cv.imshow('Video',resizedFrame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()