#!/usr/bin/python3

import cv2
import numpy as np
import pprint as pp
import time

from VideoManager import VideoManager
from PreProcessing import PreProcessing as pre
from FireDetection import *


print('OpenCV version:', cv2.__version__)



''' Image Acquisition '''
vm = VideoManager()

vm.addStream('rgb', 'media/rgb.mp4')
vm.addStream('swir', 'media/swir.mp4')

# img = cv2.imread('media/flame.png')

# Time measuring snippit
# t0 = time.time()
# print('Preprocessing: ',round(time.time() - t0, 3), 's')

frames = vm.getFrames()
i = 0
while(i < 200):
	name = 'swir'
	frames[name] = vm.getFrame(name)
	i += 1

key = 0
wait = 1
scale = 0.35
while(key != ord('q')):
	frames = vm.getFrames()

	if not frames:
		break

	high_x = [0,'']
	high_y = [0,'']
	for frame in frames:
		if frames[frame].shape[0] > high_x[0]:
			high_x[0] = frames[frame].shape[0]
			high_x[1] = frame
		if frames[frame].shape[1] > high_y[0]:
			high_y[0] = frames[frame].shape[1]
			high_y[1] = frame

	for frame in frames:
		if frame == 'rgb':
			if frames[frame].shape[0] > frames[frame].shape[1]:
				# Resize so the image isn't too big could do this at the end - camera(stream) specific
				rgb_res = pre.scale(frames[frame], high_x[0]/frames[frame].shape[0]*scale)
			else:
				# Resize so the image isn't too big could do this at the end - camera(stream) specific
				rgb_res = pre.scale(frames[frame], high_y[0]/frames[frame].shape[1]*scale)

			# Initialize Fire detection module
			FD = RGBDetector()
			# Detect flames on a RGB image using contour mode
			rgb_det, rgb_mask = FD.detectFlames(rgb_res, 'contours')

			frames[frame] = rgb_det
			# # Combine the two images for better comparison
			# comp = np.concatenate((rgb_det,rgb_mask),axis=1)
			# cv2.imshow('compare',comp)

			# # Check for key input
			# key = cv2.waitKey(wait) & 0xFF
			# wait = 1
			# # Pause the program if the key 'p' is pressed
			# if key == ord('p'):
			# 	wait = 0

		if frame == 'flir':
			''' Preprocessing '''
			# Cut off level indicator bar
			frames[frame] = frames[frame][0:frames[frame].shape[0]-30,0:frames[frame].shape[1]]

			if frames[frame].shape[0] > frames[frame].shape[1]:
				# Resize so the image isn't too big could do this at the end - camera(stream) specific
				flir_res = pre.scale(frames[frame], high_x[0]/frames[frame].shape[0]*scale)
			else:
				# Resize so the image isn't too big could do this at the end - camera(stream) specific
				flir_res = pre.scale(frames[frame], high_y[0]/frames[frame].shape[1]*scale)
			frames[frame] = flir_res

		if frame == 'swir':
			''' Preprocessing '''
			# Cut off level indicator bar
			# frames[frame] = frames[frame][0:frames[frame].shape[0]-30,0:frames[frame].shape[1]]

			if frames[frame].shape[0] > frames[frame].shape[1]:
				# Resize so the image isn't too big could do this at the end - camera(stream) specific
				swir_res = pre.scale(frames[frame], high_x[0]/frames[frame].shape[0]*scale)
			else:
				# Resize so the image isn't too big could do this at the end - camera(stream) specific
				swir_res = pre.scale(frames[frame], high_y[0]/frames[frame].shape[1]*scale)
			frames[frame] = swir_res




	# Create white picture grayscale for the purpose of countour/blob detection
	blank = np.full(frames['rgb'].shape, 0, np.uint8)
	blank[0:frames['swir'].shape[0],0:frames['swir'].shape[1]] = frames['swir']
	comp = np.concatenate((frames['rgb'],blank),axis=1)
	# comp = pre.scale(comp, 0.35)
	cv2.imshow('compare',comp)
	# # Check for key input
	key = cv2.waitKey(wait) & 0xFF
	wait = 1
	# Pause the program if the key 'p' is pressed
	if key == ord('p'):
		wait = 0

''' Closing the video streams '''
vm.close()
cv2.destroyAllWindows()