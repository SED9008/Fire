import cv2
import numpy as np
from Colors import bgr

bgr = bgr()

class FireDetector:
	def __init__(self, image_type):
		self.image_type = image_type
		if 	image_type == 'color':
			# Lower and upper HSV mask boundaries for detecting flame pixels
			self.lower_h = 1
			self.lower_s = 220
			self.lower_v = 230

			self.upper_h = 50
			self.upper_s = 255
			self.upper_v = 255

			# Minimal area to avoid false positives (noise)
			self.area_blob 		= 3
			self.area_contour 	= 100

		elif image_type == 'swir':
			# Threshold values for detecting hot pixels
			self.lower = 220
			self.upper = 255

			# Minimal area to avoid false positives (noise)
			self.area_blob = 2
			self.area_contour = 200

		elif image_type == 'flir':
			# Threshold values for detecting hot pixels
			self.lower = 220
			self.upper = 255

			# Minimal area to avoid false positives (noise)
			self.area_blob = 2
			self.area_contour = 200

		else:
			print('Unknown video type')
		
	def detect(self, img):
		fire = False

		if self.image_type == 'color':

			# Convert the BGR image to HSV
			img_hsv 	= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			# Apply a threshold on all channels
			mask_hsv 	= cv2.inRange(img_hsv, np.array([self.lower_h,self.lower_s,self.lower_v]), np.array([self.upper_h,self.upper_s,self.upper_v]))
			# Mask the area on the RGB image
			masked 		= cv2.bitwise_and(img, img, mask = mask_hsv)
			# Create white picture grayscale for the purpose of countour/blob detection
			blank 		= np.full(mask_hsv.shape, 255, np.uint8)
			# Create grayscale mask for the purpose of contour/blob detection
			mask_gray 	= cv2.bitwise_and(blank, blank, mask = mask_hsv)
			# Detect possible fire contours using an area threshold
			contours 	= self.getContours(mask_gray)

			if len(contours) > 0:
				# Draw a rectangle around the contours
				img 	= self.drawContours(img,contours)
				fire 	= True
				
			return fire, img, masked

		elif self.image_type == 'swir' or self.image_type == 'flir':
			# SWIR is already a grayscale image but openCV reads it as a BGR image
			img_gray 	= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# Apply a threshold 
			mask_gray 	= cv2.inRange(img_gray, self.lower, self.upper)
			# Mask the area on the RGB image
			masked 	= cv2.bitwise_and(img, img, mask = mask_gray)
			# Detect possible fire contours using an area threshold
			contours 	= self.getContours(mask_gray)

			if len(contours) > 0:
				# Draw a rectangle around the contours
				img 	= self.drawContours(img,contours)
				fire 	= True
				
			return fire, img, masked
			
		else:
			print('Unknown video type')

	def getContours(self, img):
		# Detect contours
		img_cnt, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		filtered_contours = []

		# Check if any contours are detected
		if len(contours) > 0:
			for contour in contours:
				# Apply area threshold to avoid false positives (noise)
				area = cv2.contourArea(contour)
				if area > self.area_contour:
					x,y,w,h = cv2.boundingRect(contour)
					filtered_contours.append((x,y,w,h))

		return filtered_contours

	def drawContours(self, img, contours):
		for contour in contours:
			x 		= contour[0]
			y 		= contour[1]
			w 		= contour[2]
			h 		= contour[3]
			img 	= cv2.rectangle(img, 	(x-int(w/2),y-int(h/2)),(x+w,y+h),bgr.green,2)

		return img