#!/usr/bin/python3

'''
Webcam example
Needs correction dynamic saturation and white balance
'''

import cv2
import numpy as np
import time

from VideoManager import VideoManager
from PreProcessing import PreProcessing as pre
from FireDetection import *
from Colors import bgr

# Some bgr color code simplifications
bgr = bgr()

SCALE = 1

streams = 	{	
			'color':{		
				'filename':0,
				'scale':SCALE,
				'offset':0,
					}
			}

# Create the videomanager
vm = VideoManager()

detectors = {}

for stream in streams:
	# Add a videostream
	vm.addStream(stream, streams[stream]['filename'])
	# Create the corrosponding detectors
	detectors[stream] = FireDetector(stream)

run 	= True
wait 	= 1
fire 	= False
mask	= 0

while(run):
	for stream in streams:
		ret, streams[stream]['frame'] = vm.getFrame(stream)

		fire, streams[stream]['frame'], streams[stream]['mask']	= detectors[stream].detect(streams[stream]['frame'])

		if fire:
			text_color 	= bgr.red 
		else:
			text_color 	= bgr.white

		cv2.putText(streams[stream]['frame'], stream, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
		cv2.putText(streams[stream]['frame'], stream, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)
		cv2.putText(streams[stream]['mask'], stream+' mask', (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
		cv2.putText(streams[stream]['mask'], stream+' mask', (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)

	if ret:
		blank = np.full(((streams['color']['frame'].shape[0]*mask)+streams['color']['frame'].shape[0],streams['color']['frame'].shape[1]*len(streams),3), 0, np.uint8)

		blank[0:streams['color']['frame'].shape[0],0:streams['color']['frame'].shape[1]] = streams['color']['frame']

		if mask:
			blank[streams['color']['frame'].shape[0]:streams['color']['frame'].shape[0]*2,0:streams['color']['frame'].shape[1]] = streams['color']['mask']

		cv2.imshow('Fire detection webcam',blank)

		key = cv2.waitKey(wait) & 0xFF
		if key == ord('q'):
				run = False	
		elif key == ord('m'):
				mask = 1	

vm.close()
cv2.destroyAllWindows()