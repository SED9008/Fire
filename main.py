#!/usr/bin/python3

# ffmpeg -i rgb_kelder.mp4 rgb_kelder-%04d.png
# ffmpeg -framerate 1.23101 -i rgb_kelder-%04d.png -r 1.23101 rgb_kelder_ori.mp4
# ffmpeg -r 1.23101 -i rgb_kelder_ori.mp4 -r 1 rgb_kelder_1fps.mp4

import cv2
import numpy as np
import pprint as pp
import time

from VideoManager import VideoManager
from PreProcessing import PreProcessing as pre
from FireDetection import *
from Colors import bgr

bgr = bgr()

SCALE = 0.2

# Specify streams

videosets	=	{
				'slaapkamer': 	{	
								'rgb':{		
									'filename':'media/rgb_slaapkamer_1fps.mp4',
									'scale':SCALE,
									'offset':34,
									}, 
								'swir':{	
									'filename':'media/swir_slaapkamer_1fps.mp4',
									'scale':SCALE * 3.2,
									'offset':15,
									},
								'flir':{
									'filename':'media/flir_slaapkamer_1fps.mp4',
									'scale':SCALE * 1.6,
									'offset':0,
									},
								},
				'kelder':		{	
								'rgb':{		
									'filename':'media/rgb_kelder_1fps.mp4',
									'scale':SCALE,
									'offset':8,
									}, 
								'swir':{	
									'filename':'media/swir_kelder_1fps.mp4',
									'scale':SCALE * 3.2,
									'offset':3,
									},
								'flir':{
									'filename':'media/flir_kelder_1fps.mp4',
									'scale':SCALE * 1.6,
									'offset':0,
									},
								}
				}

# Initial values
wait 			= 1
speed 			= 1
mask_overlay	= False
run 			= True

while(run):
	for videoset in videosets:
		streams = videosets[videoset]
		# Create the videomanager
		vm = VideoManager()

		# Sync up streams by skipping frames
		for stream in streams:
			vm.addStream(stream, streams[stream]['filename'])
			vm.setFrameIndex(stream, streams[stream]['offset'])

		# Initialize detectors
		RGB 	= RGBDetector()
		SWIR 	= SWIRDetector()
		FLIR 	= SWIRDetector()


		frame_index 	= 1

		while(run):
			for stream in streams:
				
				ret, streams[stream]['frame'] 	= vm.getFrame(stream) 

				if ret:
					if stream == 'rgb':
						streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
						# Detect flames on a RGB image using contour mode
						flame, rgb_det, rgb_mask 	= RGB.detectFlames(streams[stream]['frame'], stream,'contours')
						streams[stream]['frame'] 	= rgb_det

					elif stream == 'swir':
						streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
						# Detect heat on a RGB image using contour mode
						heat, swir_det, swir_mask 	= SWIR.detectHeat(streams[stream]['frame'], stream, 'contours')
						streams[stream]['frame'] 	= swir_det

					elif stream == 'flir':
						streams[stream]['frame']	= pre.scale(streams[stream]['frame'], streams[stream]['scale'])
						# Cut off level indicator bar
						streams[stream]['frame'] 	= streams[stream]['frame'][0:streams[stream]['frame'].shape[0]-30,0:streams[stream]['frame'].shape[1]]
						# Detect heat on a RGB image using contour mode
						heat, flir_det, flir_mask 	= FLIR.detectHeat(streams[stream]['frame'], stream,'contours')
						streams[stream]['frame'] 	= flir_det
				else:
					break

			if ret:
				frame_index += 1

				if mask_overlay:
					blank = np.full((streams['rgb']['frame'].shape[0]*2,streams['rgb']['frame'].shape[1]*3,3), 0, np.uint8)
					blank[streams['rgb']['frame'].shape[0]:streams['rgb']['frame'].shape[0]*2,0:streams['rgb']['frame'].shape[1]] 																		= rgb_mask
					blank[streams['rgb']['frame'].shape[0]:streams['swir']['frame'].shape[0]+streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[1]:streams['swir']['frame'].shape[1]*2] 	= swir_mask
					blank[streams['rgb']['frame'].shape[0]:streams['flir']['frame'].shape[0]+streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[1]*2:streams['flir']['frame'].shape[1]*3] 	= flir_mask
					cv2.putText(blank, 'Speed: '+ str(round((1/speed)*1000)),	(blank.shape[1]-160,int(blank.shape[0]/2)-15),	cv2.FONT_HERSHEY_SIMPLEX,0.75,bgr.white,2,cv2.LINE_AA)
					cv2.rectangle(blank,(streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[0]+1),	(streams['rgb']['frame'].shape[0]*2,int(blank.shape[0]+1)),		bgr.white,	2)
					test = 1
				else:
					blank = np.full((streams['rgb']['frame'].shape[0],streams['rgb']['frame'].shape[1]*3,3), 0, np.uint8)
					cv2.putText(blank, 'Speed: '+ str(round((1/speed)*1000)), (blank.shape[1]-160,blank.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX,0.75,bgr.white,2,cv2.LINE_AA)
				
				blank[0:streams['rgb']['frame'].shape[0],0:streams['rgb']['frame'].shape[1]] 										= streams['rgb']['frame']
				blank[0:streams['swir']['frame'].shape[0],streams['rgb']['frame'].shape[1]:streams['swir']['frame'].shape[1]*2] 	= streams['swir']['frame']
				blank[0:streams['flir']['frame'].shape[0],streams['rgb']['frame'].shape[1]*2:streams['flir']['frame'].shape[1]*3] 	= streams['flir']['frame']
				
				cv2.rectangle(blank,(streams['rgb']['frame'].shape[0],-2),	(streams['rgb']['frame'].shape[0]*2,int(streams['rgb']['frame'].shape[0])+1), bgr.white, 2)
				# Middle line change this to a line! no rectangle necessary
				cv2.rectangle(blank,(-2,streams['rgb']['frame'].shape[0]+1),(blank.shape[1]+1,blank.shape[0]+1), bgr.white,	2)

				cv2.imshow('Fire detection',blank)

				# Section that handles al different key presses
				key = cv2.waitKey(wait) & 0xFF

				if key == ord('v'):
					break
				if key == ord('p'):
					wait = 0
				elif key == ord('m'):
					mask_overlay = not mask_overlay	
				elif key == ord('q'):
					run = False	
				elif key == ord(','):
					#slower
					if wait < 1000:
						speed 	= speed*2
						wait 	= speed
				elif key == ord('.'):
					#faster
					if wait > 1:
						speed 	= int(speed/2)
						wait 	= speed
				elif key == 81:
					if frame_index > 1:
						frame_index -= 2
					for stream in streams:
						# Very slow .. don't know why
						vm.setFrameIndex(stream, streams[stream]['offset']+frame_index)
					wait = 0
				elif key == 83:
					wait = 0
				else:
					wait = speed
			else:
				break

vm.close()
cv2.destroyAllWindows()