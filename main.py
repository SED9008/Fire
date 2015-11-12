#!/usr/bin/python3

import cv2
import numpy as np
import pprint
import time

from VideoManager import VideoManager


print('OpenCV version:', cv2.__version__)



''' Image Acquisition '''
vm = VideoManager()

vm.addStream('rgb', 'media/rgb.mp4')
#vm.addStream('swir')

# frames = {}
# frames['rgb2'] = vm.getFrame('rgb2')

# img = cv2.imread('media/flame.png')

# Time measuring snippit
# t0 = time.time()
# print('Preprocessing: ',round(time.time() - t0, 3), 's')

key = 0

while(key != 113):
	frames = vm.getFrames()
	if not frames:
		break
	for frame in frames:
		
		if frame == 'rgb':
			''' Image Preprocessing '''
			# Resize so the image isn't too big - camera(stream) specific 
			res = cv2.resize(frames[frame], None, fx=0.35, fy=0.35, interpolation = cv2.INTER_CUBIC)

			''' Feature extraction '''
			# Convert to HSV color space
			hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
			# Create the mask so only yellow and red and high saturation and values are passed (qualities of fire)
			mask_hsv = cv2.inRange(hsv, np.array([5,220,240]), np.array([35,255,255]))
			# Mask the area on the RGB image
			mask_rgb = cv2.bitwise_and(res, res, mask = mask_hsv)
			# Combine the two images for better comparison
			comp = np.concatenate((res,mask_rgb),axis=1)
		
		cv2.imshow('compare',comp)
		key = cv2.waitKey(20) & 0xFF


''' Closing the video streams '''
vm.close()
cv2.destroyAllWindows()