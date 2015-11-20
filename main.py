#!/usr/bin/python3

import cv2
import numpy as np
import pprint as pp
import time

from VideoManager import VideoManager
from PreProcessing import PreProcessing as pre
from FireDetection import *

SCALE = 0.26

# Specify streams
streams = {	'rgb':{		
					'filename':'media/rgb_1fps.mp4',
					'scale':SCALE,
					'offset':34,
					}, 
			'swir':{	
					'filename':'media/swir_1fps.mp4',
					'scale':SCALE * 3.2,
					'offset':15,
					},
			'flir':{
					'filename':'media/flir_1fps.mp4',
					'scale':SCALE * 1.6,
					'offset':0,
					},
			}

vm = VideoManager()

for stream in streams:
	vm.addStream(stream, streams[stream]['filename'])
	vm.skipFrame(stream, streams[stream]['offset'])
	streams[stream]['frame'] = vm.getFrame(stream)
	streams[stream]['frame'] = pre.scale(streams[stream]['frame'], streams[stream]['scale'])


# Initial values
wait = 1
while(True):
	for stream in streams:
		streams[stream]['frame'] 	= vm.getFrame(stream) 
		

		if stream == 'rgb':
			streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])

			# Initialize Fire detection module
			RGB = RGBDetector()
			# Detect flames on a RGB image using contour mode
			rgb_det, rgb_mask = RGB.detectFlames(streams[stream]['frame'], 'contours')

			streams[stream]['frame'] = rgb_det

		elif stream == 'swir':
			streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
			# Initialize Fire detection module
			SWIR = SWIRDetector()
			# Detect heat on a RGB image using contour mode
			swir_det, swir_mask = SWIR.detectHeat(streams[stream]['frame'] , 'contours')

			streams[stream]['frame'] = swir_det
			
		elif stream == 'flir':
			streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
			# Cut off level indicator bar
			streams[stream]['frame'] = streams[stream]['frame'][0:streams[stream]['frame'].shape[0]-30,0:streams[stream]['frame'].shape[1]]

	blank = np.full((streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[1]*2,3), 0, np.uint8)
	blank[0:streams['swir']['frame'].shape[0],0:streams['swir']['frame'].shape[1]] = streams['swir']['frame']
	blank[0:streams['flir']['frame'].shape[0],streams['swir']['frame'].shape[1]:streams['flir']['frame'].shape[1]*2] = streams['flir']['frame']
	comp = np.concatenate((streams['rgb']['frame'],blank),axis=1)
	cv2.imshow('test',comp)
	key = cv2.waitKey(wait) & 0xFF
	if key == ord('p'):
		wait = 0
	elif key == ord('q'):
		break	
	else:
		wait = 1



# while(True):

# 	test_frame = vm.getFrame('swir')



# 	cv2.imshow('test',test_frame)

# 	key = cv2.waitKey(wait) & 0xFF

# 	wait = default_wait / speed

# 	if key == ord('p'):
# 		wait = 0
# 	if key == ord('q'):
# 		break

vm.close()
cv2.destroyAllWindows()