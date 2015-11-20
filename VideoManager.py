import cv2

class VideoManager:
	def __init__(self):
		self.videos = {}

	def addStream(self, name, filename):
		self.videos[name] = cv2.VideoCapture(filename)
		if self.videos[name]:
			print('Opened video:', name)
		else:
			print('Something went wrong while opening', name)

	def getFrame(self, video):
		ret, frame = self.videos[video].read()
		if not ret:
			print('Error in retreiving frame from:', video)
			return False
		return frame

	def skipFrame(self, video, amount):
		print('Skipping', amount, video,'frames')
		for i in range(0,amount):
			self.videos[video].grab()
		ret, frame = self.videos[video].read()
		return frame


	def close(self):
		for video in self.videos:
			self.videos[video].release()
			print('Closed video:', video)