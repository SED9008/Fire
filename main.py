#!/usr/bin/python3

import cv2
import pprint


print("OpenCV version:", cv2.__version__)

filename = "rgb.mp4"

video = cv2.VideoCapture(filename)

while(true):
	# Capture frame-by-frame
    ret, frame = video.read()

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()