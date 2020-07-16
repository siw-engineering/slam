import cv2
import glob
import re

"""
Loads video files
"""
class Video(object):
	def __init__(self, path, preprocessor=None):
		cap = cv2.VideoCapture(path)
		self.path = path
		self.cap = cap
		self.preprocessor = preprocessor

	def __len__(self):
		return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	@property
	def w(self):
		return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	
	@property
	def h(self):
		return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	@property
	def frame(self):
		_, _frame = self.cap.read()
		if self.preprocessor:
			_frame = self.preprocessor(_frame)
		return _frame