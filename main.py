#!/usr/bin/python3

import cv2
import pprint

from VideoManager import VideoManager


print('OpenCV version:', cv2.__version__)


vm = VideoManager()

vm.addStream('rgb2')
vm.addStream('swir')

frames = vm.getFrames()
for frame in frames:
	cv2.imshow(frame,frames[frame])
	cv2.waitKey(0)

vm.close()
cv2.destroyAllWindows()