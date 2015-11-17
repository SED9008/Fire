import cv2

class VideoManager:
	def __init__(self):
		self.videos = {}

	def addStream(self, name, filename):
		self.videos[name] = cv2.VideoCapture(filename)
		print(self.videos[name].get(5))

	def getFrame(self, video):
		ret, frame = self.videos[video].read()
		if not ret:
			print('Error in retreiving frame from:', video)
			return False
		return frame

	def getFrames(self):
		frames = {}
		for video in self.videos:
			ret, frames[video] = self.videos[video].read()
			if not ret:
				print('Error in retreiving frame from:', video)
				return False
		return frames

	def close(self):
		for video in self.videos:
			self.videos[video].release()
			print('Closed ', video)