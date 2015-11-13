#!/usr/bin/python3

import cv2
import numpy as np
import pprint as pp
import time

from VideoManager import VideoManager
from PreProcessing import PreProcessing as pre
from FireDetection import FireDetection


print('OpenCV version:', cv2.__version__)



''' Image Acquisition '''
vm = VideoManager()

vm.addStream('rgb', 'media/rgb.mp4')
#vm.addStream('swir', 'media/swir.mp4')

# img = cv2.imread('media/flame.png')

# Time measuring snippit
# t0 = time.time()
# print('Preprocessing: ',round(time.time() - t0, 3), 's')

key = 0
wait = 1
while(key != ord('q')):
	frames = vm.getFrames()
	if not frames:
		break
	for frame in frames:
		
		if frame == 'rgb':
			# Resize so the image isn't too big could do this at the end - camera(stream) specific
			rgb_res = pre.scale(frames[frame], 0.35)

			# Initialize Fire detection module
			FD = FireDetection()
			# Detect flames on a RGB image using contour mode
			rgb_det, rgb_mask = FD.detectFlamesRGB(rgb_res, 'contours')

			# Combine the two images for better comparison
			comp = np.concatenate((rgb_det,rgb_mask),axis=1)
			cv2.imshow('compare',comp)

			# Check for key input
			key = cv2.waitKey(wait) & 0xFF
			wait = 1
			# Pause the program if the key 'p' is pressed
			if key == ord('p'):
				wait = 0

''' Closing the video streams '''
vm.close()
cv2.destroyAllWindows()