import cv2
import numpy as np
from Colors import bgr

bgr = bgr()

class RGBDetector:
	def __init__(self):
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

	def detectFlames(self, img, name, mode):
		''' Pre Processing '''
		img_hsv 	= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		''' Feature extraction '''
		# # Create the mask so only yellow and red and high saturation and values are passed (qualities of fire)
		mask_hsv 	= cv2.inRange(img_hsv, np.array([self.lower_h,self.lower_s,self.lower_v]), np.array([self.upper_h,self.upper_s,self.upper_v]))
		# Mask the area on the RGB image
		masked 		= cv2.bitwise_and(img, img, mask = mask_hsv)
		# Create white picture grayscale for the purpose of countour/blob detection
		blank 		= np.full(mask_hsv.shape, 255, np.uint8)
		# Create grayscale mask for the purpose of contour/blob detection
		mask_gray 	= cv2.bitwise_and(blank, blank, mask = mask_hsv)

		''' Detection '''
		# Fire bool for text and later on high lvl decision making
		fire = False

		# Blob detection on the HSV fire regions
		if mode == 'blobs':
			blob_params = cv2.SimpleBlobDetector_Params()
			blob_params.filterByArea = True
			blob_params.minArea = self.area_blob

			detector = cv2.SimpleBlobDetector_create(blob_params)

			keypoints = detector.detect(mask_gray)

			if len(keypoints) > 0:
				for keypoint in keypoints:
					# Apply area threshold to avoid false positives (noise)
					if keypoint.size > self.area_blob:
						fire = True
						# Get the position of the blob
						x,y = keypoint.pt
						# Draw a circle around the blob on the original and the mask image
						img = cv2.circle(img, (int(x),int(y)), int(keypoint.size)*3, bgr.green, 5)
						masked = cv2.circle(masked, (int(x),int(y)), int(keypoint.size)*3, bgr.green, 5)

		# Contour detection on the HSV fire regions
		elif mode == 'contours':
			# Detect contours
			img_cnt, contours, hierarchy = cv2.findContours(mask_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
			if len(contours) > 0:
				for contour in contours:
					# Apply area threshold to avoid false positives (noise)
					if cv2.contourArea(contour) > self.area_contour:
						fire 	= True
						# Get the position and size of the contour bounding box
						x,y,w,h = cv2.boundingRect(contour)
						# Draw the boundingboxes on the original and the mask image
						img 	= cv2.rectangle(img, 	(x-int(w/2),y-int(h/2)),(x+w,y+h),bgr.green,2)
						masked 	= cv2.rectangle(masked, (x-int(w/2),y-int(h/2)),(x+w,y+h),bgr.green,2)
		
		if fire:
			text_color = bgr.red
		else:
			text_color = bgr.white

		cv2.putText(img, name, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
		cv2.putText(img, name, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)
		cv2.putText(masked, name+' mask',(10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
		cv2.putText(masked, name+' mask',(10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)	
		
		return fire, img, masked

class SWIRDetector:
	def __init__(self):
		self.lower = 220
		self.higher = 255

		self.area_blob = 2
		self.area_contour = 200

	def detectHeat(self, img, name, mode):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		''' Feature extraction '''
		# Create the mask so only yellow and red and high saturation and values are passed (qualities of fire)
		mask 	= cv2.inRange(img_gray, self.lower, self.higher)
		# Mask the area on the RGB image
		masked 	= cv2.bitwise_and(img, img, mask = mask)

		''' Detection '''
		# Fire bool for text and later on high lvl decision making
		fire = False

		# Blob detection on the HSV fire regions
		if mode == 'blobs':
			blob_params = cv2.SimpleBlobDetector_Params()
			# Filter by Area.
			blob_params.filterByArea 	= True
			blob_params.minArea 		= self.area_blob
			# print(blob_params)
			detector 	= cv2.SimpleBlobDetector_create(blob_params)
			keypoints 	= detector.detect(mask)
			if len(keypoints) > 0:
				for keypoint in keypoints:
					print(keypoint.size)
					# Apply area threshold to avoid false positives (noise)
					if keypoint.size > self.area_blob:
						fire 	= True
						# Get the position of the blob
						x,y 	= keypoint.pt
						# Draw a circle around the blob on the original and the mask image
						img 	= cv2.circle(img, 		(int(x),int(y)), int(keypoint.size)*3, bgr.green, 5)
						masked 	= cv2.circle(masked, 	(int(x),int(y)), int(keypoint.size)*3, bgr.green, 5)

		# Contour detection on the HSV fire regions
		elif mode == 'contours':
			# Detect contours
			img_cnt, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
			if len(contours) > 0:
				for contour in contours:
					# Apply area threshold to avoid false positives (noise)
					if cv2.contourArea(contour) > self.area_contour:
						fire 	= True
						# Get the position and size of the contour bounding box
						x,y,w,h = cv2.boundingRect(contour)
						# Draw the boundingboxes on the original and the mask image
						img 	= cv2.rectangle(img, 	(x,y),(int(x+(w*1)),int(y+(h*1))),bgr.green,2)
						masked 	= cv2.rectangle(masked, (x,y),(int(x+(w*1)),int(y+(h*1))),bgr.green,2)

		if fire:
			text_color = bgr.red
		else:
			text_color = bgr.white

		cv2.putText(img, name, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
		cv2.putText(img, name, (10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)
		cv2.putText(masked, name+' mask',(10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,bgr.black,6,cv2.LINE_AA)
		cv2.putText(masked, name+' mask',(10,30), cv2.FONT_HERSHEY_DUPLEX,0.9,text_color,2,cv2.LINE_AA)	

		return fire, img, masked