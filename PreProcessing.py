import cv2

class PreProcessing:
	def scale(img, factor):
		return cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)