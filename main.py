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

# Some bgr color code simplifications
bgr = bgr()

# Subjective scale value
SCALE = 0.2

# Specify streams
videosets	=	{
				'slaapkamer': 	{	
								'color':{		
									'filename':'media/color_slaapkamer_1fps.mp4',
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
								'color':{		
									'filename':'media/color_kelder_1fps.mp4',
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

# Select a videoset
streams = videosets['kelder']

# Create the videomanager
vm = VideoManager()

detectors = {}

# Sync up streams by changing the frameindex to certain streams 
for stream in streams:
	# Add a videostream
	vm.addStream(stream, streams[stream]['filename'])
	# Set the frame index for syncing purposes
	vm.setFrameIndex(stream, streams[stream]['offset'])
	# Create the corrosponding detectors
	detectors[stream] = FireDetector(stream)



run 	= True
mask 	= False
wait 	= 1
speed 	= 1

while(run):
	for stream in streams:
		ret, streams[stream]['frame'] = vm.getFrame(stream)

		if ret:
			if stream == 'flir':
				# Cut off level indicator bar
				streams[stream]['frame'] = streams[stream]['frame'][0:streams[stream]['frame'].shape[0]-30,0:streams[stream]['frame'].shape[1]]
			# Scale images to a reasonable size
			streams[stream]['frame'] = pre.scale(streams[stream]['frame'], streams[stream]['scale'])
			# Detect fire, returns the mask for development purposes
			fire, streams[stream]['frame'], streams[stream]['mask']	= detectors[stream].detect(streams[stream]['frame'])

			if fire:
				text_color 	= bgr.red 
			else:
				text_color 	= bgr.white

			cv2.putText(streams[stream]['frame'], stream, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
			cv2.putText(streams[stream]['frame'], stream, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)
			cv2.putText(streams[stream]['mask'], stream+' mask', (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
			cv2.putText(streams[stream]['mask'], stream+' mask', (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)

		else: 
			run = False

	if ret:
		blank = np.full(((streams['color']['frame'].shape[0]*mask)+streams['color']['frame'].shape[0],streams['color']['frame'].shape[1]*len(streams),3), 0, np.uint8)

		blank[0:streams['color']['frame'].shape[0],0:streams['color']['frame'].shape[1]] 										= streams['color']['frame']
		blank[0:streams['swir']['frame'].shape[0],streams['color']['frame'].shape[1]:streams['swir']['frame'].shape[1]*2] 	= streams['swir']['frame']
		blank[0:streams['flir']['frame'].shape[0],streams['color']['frame'].shape[1]*2:streams['flir']['frame'].shape[1]*3] = streams['flir']['frame']



		cv2.rectangle(blank,(0,0),(streams['color']['frame'].shape[0]*len(streams)-1,streams['color']['frame'].shape[0]-1),bgr.white,1)
		cv2.rectangle(blank,(streams['color']['frame'].shape[0],0),(streams['color']['frame'].shape[0]*2-1,streams['color']['frame'].shape[0]-1),bgr.white,1)

		if mask:
			blank[streams['color']['frame'].shape[0]:streams['color']['frame'].shape[0]*2,0:streams['color']['frame'].shape[1]] 																		= streams['color']['mask']
			blank[streams['color']['frame'].shape[0]:streams['swir']['frame'].shape[0]+streams['color']['frame'].shape[0],streams['color']['frame'].shape[1]:streams['swir']['frame'].shape[1]*2] 		= streams['swir']['mask']
			blank[streams['color']['frame'].shape[0]:streams['flir']['frame'].shape[0]+streams['color']['frame'].shape[0],streams['color']['frame'].shape[1]*2:streams['flir']['frame'].shape[1]*3] 	= streams['flir']['mask']
			
			cv2.rectangle(blank,(0,streams['color']['frame'].shape[0]-1),(streams['color']['frame'].shape[0]*len(streams)-1,streams['color']['frame'].shape[0]*2-1),bgr.white,1)
			cv2.rectangle(blank,(streams['color']['frame'].shape[0],streams['color']['frame'].shape[0]-1),(streams['color']['frame'].shape[0]*2-1,streams['color']['frame'].shape[0]*2-1),bgr.white,1)

		cv2.imshow('Fire detection',blank)

		key = cv2.waitKey(wait) & 0xFF

		if key == ord('p'):
			wait = 0
		elif key == ord('m'):
			mask = not mask
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
		elif key == 83:
			wait = 0
		else:
			wait = speed


vm.close()
cv2.destroyAllWindows()