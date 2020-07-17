import sys
sys.path.append("/home/developer/slam/slam/sensors")
from buffer import Buffer
import numpy as np

class Camera:
	_buf = None
	def __init__(self, F=None, stream=None, l=100):
		self.F = F
		self.stream = stream
		self._buf = Buffer(l=l)
		self.stream.start(self._buf)

	def read(self):
		data = self._buf.read()
		return data

	def has_next(self):
		return self._buf.has_next()

	@property
	def K(self):
		return self._K

	@property
	def Kinv(self):
		return self._Kinv
	