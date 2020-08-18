import cv2
import os


class TumStream(object):
	def __init__(self, path, preprocess=True):
		self.path = path
		self.depth_file = open(os.path.join(self.path, "depth.txt")).readlines()
		self.rgb_file = open(os.path.join(self.path, "rgb.txt")).readlines()
		self.ptr = 0

	def __len__(self):
		pass

	def rgb(self, ptr):
		file = self.rgb_file[ptr].split(" ")[1].strip()
		_frame = cv2.imread(os.path.join(self.path, file))
		return _frame

	def depth(self, ptr):
		file = self.depth_file[ptr].split(" ")[1].strip()
		_frame = cv2.imread(os.path.join(self.path, file))
		return _frame

	def rgbd(self):
		if self.ptr > len(self.rgb_file) and self.ptr >len(self.depth_file):
			return None
		else:
			self.ptr += 1
			return self.rgb(self.ptr), self.depth(self.ptr)

