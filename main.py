#!/usr/bin/python3

import cv2
import numpy as np
import pprint as pp
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
			# Resize so the image isn't too big could do this at the end - camera(stream) specific 
			res = cv2.resize(frames[frame], None, fx=0.35, fy=0.35, interpolation = cv2.INTER_CUBIC)
			# Blur the source image to reduce color noise 
        	# cv.Smooth(img, img, cv.CV_BLUR, 3); 

			''' Feature extraction '''
			# Convert to HSV color space
			hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
			# Create the mask so only yellow and red and high saturation and values are passed (qualities of fire)
			mask_hsv = cv2.inRange(hsv, np.array([5,220,240]), np.array([35,255,255]))
			# Mask the area on the RGB image
			masked_rgb = cv2.bitwise_and(res, res, mask = mask_hsv)

			# Create white picture grayscale for the purpose of countour/blob detection
			blank = np.full(mask_hsv.shape, 255, np.uint8)
			mask_gray = cv2.bitwise_and(blank, blank, mask = mask_hsv)

			blob_params = cv2.SimpleBlobDetector_Params()
			# Filter by Area.
			blob_params.filterByArea = True
			blob_params.minArea = 3
			# print(blob_params)
			detector = cv2.SimpleBlobDetector_create(blob_params)
			keypoints = detector.detect(mask_gray)
			if len(keypoints) > 0:
				points = []
				for keypoint in keypoints:
					points.append((keypoint.size, keypoint.pt))
				points.sort(reverse=True)

				res_blobs = cv2.circle(res, (int(points[0][1][0]),int(points[0][1][1])), int(points[0][0])*3, (0,255,0), 5)
				masked_rgb_blobs = cv2.circle(masked_rgb, (int(points[0][1][0]),int(points[0][1][1])), int(points[0][0])*3, (0,255,0), 5)
				comp = np.concatenate((res_blobs,masked_rgb_blobs),axis=1)
			else:
				comp = np.concatenate((res,masked_rgb),axis=1)
			
			
			# res_blobs = cv2.drawKeypoints(res, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			# mask_gray, contours, hierarchy = cv2.findContours(mask_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

			# contours_drawn = cv2.drawContours(res, contours, -1, (0,255,0), 10)

		# Combine the two images for better comparison
		# comp = np.concatenate((res_blobs,masked_rgb),axis=1)
		cv2.imshow('compare',comp)
		key = cv2.waitKey(1) & 0xFF


''' Closing the video streams '''
vm.close()
cv2.destroyAllWindows()