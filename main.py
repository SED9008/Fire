#!/usr/bin/python3

import cv2
import pprint

from VideoManager import VideoManager


print('OpenCV version:', cv2.__version__)


# # Video acquisition
# # Add video names
# videos = {
# 			'rgb2':''
# 			,'swir':''
# 			}

# for video in videos:
# 	videos[video] = cv2.VideoCapture(video + '.mp4')



# ret, frame = videos['rgb2'].read()


vm = VideoManager()

vm.addStream('rgb2')
vm.addStream('swir')

frames = vm.getFrames()
for frame in frames:
	cv2.imshow(frame,frames[frame])
	cv2.waitKey(0)
	

# while(1):
# 	for video in videos:	
# 		# Capture frame-by-frame
# 	    ret, frame = videos[video].read()

# 	    # Display the resulting frame
# 	    cv2.imshow('frame',frame)
# 	    if cv2.waitKey(1) & 0xFF == ord('q'):
# 	        break

# # When everything done, release the capture

vm.close()
cv2.destroyAllWindows()