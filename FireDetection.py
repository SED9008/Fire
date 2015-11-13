import cv2
import numpy as np

class FireDetection:
	def __init__(self):
		self.lower_h = 1
		self.lower_s = 50
		self.lower_v = 230

		self.upper_h = 30
		self.upper_s = 255
		self.upper_v = 255

		self.area_blob = 3
		self.area_contour = 100

	def detectFlamesRGB(self, img, mode):
		''' Pre Processing '''
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


		''' Feature extraction '''
		# # Create the mask so only yellow and red and high saturation and values are passed (qualities of fire)
		mask_hsv = cv2.inRange(img, np.array([self.lower_h,self.lower_s,self.lower_v]), np.array([self.upper_h,self.upper_s,self.upper_v]))
		# Mask the area on the RGB image
		masked_rgb = cv2.bitwise_and(img, img, mask = mask_hsv)
		# Create white picture grayscale for the purpose of countour/blob detection
		blank = np.full(mask_hsv.shape, 255, np.uint8)
		# Create grayscale mask for the purpose of contour/blob detection
		mask_gray = cv2.bitwise_and(blank, blank, mask = mask_hsv)

		fire = False
		''' Detection '''
		if mode == 'blobs':
			blob_params = cv2.SimpleBlobDetector_Params()
			# Filter by Area.
			blob_params.filterByArea = True
			blob_params.minArea = self.area_blob
			# print(blob_params)
			detector = cv2.SimpleBlobDetector_create(blob_params)
			keypoints = detector.detect(mask_gray)
			if len(keypoints) > 0:
				for keypoint in keypoints:
					if keypoint.size > area:
						fire = True
						x,y = keypoint.pt
						img = cv2.circle(img, (int(x),int(y)), int(keypoint.size)*3, (0,255,0), 5)
						masked_rgb = cv2.circle(masked_rgb, (int(x),int(y)), int(keypoint.size)*3, (0,255,0), 5)

		elif mode == 'contours':
			img_cnt, contours, hierarchy = cv2.findContours(mask_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
			if len(contours) > 0:
				for contour in contours:
					if cv2.contourArea(contour) > self.area_contour:
						fire = True
						x,y,w,h = cv2.boundingRect(contour)
						img = cv2.rectangle(img, (x-w,y-h),(x+(w*2),y+(h*2)),(0,255,0),3)
						masked_rgb = cv2.rectangle(masked_rgb, (x-w,y-h),(x+(w*2),y+(h*2)),(0,255,0),3)
						
		if fire:
			cv2.putText(img, 'Flame Detected',(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),6,cv2.LINE_AA)
			cv2.putText(masked_rgb, 'Fire Detected',(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),6,cv2.LINE_AA)	
			cv2.putText(img, 'Flame Detected',(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
			cv2.putText(masked_rgb, 'Flame Detected',(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
		
		return img, masked_rgb

