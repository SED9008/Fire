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

# Initialize detectors
RGB 	= RGBDetector()
SWIR 	= SWIRDetector()
FLIR 	= SWIRDetector()


# Initial values
wait 	= 1
speed 	= 1
overlay = False
while(True):
	for stream in streams:
		streams[stream]['frame'] 	= vm.getFrame(stream) 
		if stream == 'rgb':
			streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
			# Detect flames on a RGB image using contour mode
			rgb_det, rgb_mask = RGB.detectFlames(streams[stream]['frame'], 'contours')
			streams[stream]['frame'] = rgb_det

		elif stream == 'swir':
			streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
			# Detect heat on a RGB image using contour mode
			swir_det, swir_mask = SWIR.detectHeat(streams[stream]['frame'] , 'contours')
			streams[stream]['frame'] = swir_det

		elif stream == 'flir':
			streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
			# Cut off level indicator bar
			streams[stream]['frame'] = streams[stream]['frame'][0:streams[stream]['frame'].shape[0]-30,0:streams[stream]['frame'].shape[1]]
			# Detect heat on a RGB image using contour mode
			flir_det, flir_mask = FLIR.detectHeat(streams[stream]['frame'], 'contours')
			streams[stream]['frame'] = flir_det

	if overlay:
		blank = np.full((streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[1]*2,3), 0, np.uint8)
		blank[0:streams['swir']['frame'].shape[0],0:streams['swir']['frame'].shape[1]] = swir_mask
		# blank[0:streams['flir']['frame'].shape[0],0:streams['flir']['frame'].shape[1]] = flir_mask
		blank[0:streams['flir']['frame'].shape[0],streams['swir']['frame'].shape[1]:streams['flir']['frame'].shape[1]*2] = flir_mask
		comp = np.concatenate((rgb_mask,blank),axis=1)
		cv2.imshow('Fire detection',comp)
	else:
		blank = np.full((streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[1]*2,3), 0, np.uint8)
		blank[0:streams['swir']['frame'].shape[0],0:streams['swir']['frame'].shape[1]] = streams['swir']['frame']
		# blank[0:streams['flir']['frame'].shape[0],0:streams['flir']['frame'].shape[1]] = flir_mask
		blank[0:streams['flir']['frame'].shape[0],streams['swir']['frame'].shape[1]:streams['flir']['frame'].shape[1]*2] = streams['flir']['frame']
		comp = np.concatenate((streams['rgb']['frame'],blank),axis=1)
		cv2.imshow('Fire detection',comp)

	key = cv2.waitKey(wait) & 0xFF
	if key == ord('p'):
		wait = 0
	elif key == ord('q'):
		break	
	elif key == ord('o'):
		overlay = not overlay
	elif key == 81:
		#slower
		print('slower', wait)
		if wait < 1000:
			speed 	= speed*2
			wait 	= speed
			print(wait)
	elif key == 83:
		#faster
		print('faster', wait)
		if wait > 1:
			speed 	= int(speed/2)
			wait 	= speed
			print(wait)
	else:
		wait = speed


vm.close()
cv2.destroyAllWindows()