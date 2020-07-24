import numpy as np
import sys

class Buffer:
	def __init__(self, l=100):
		self.l = l
		self.read_ptr = 0
		self.write_ptr = 0
		self.buf = [np.zeros(1)] * l 

	def __len__(self):
		return self.l

	def write(self, data):
		self.buf[self.write_ptr] = data
		self.write_ptr = (self.write_ptr + 1)%self.l

	def read(self):
		data = self.buf[self.read_ptr]
		self.read_ptr = (self.read_ptr + 1)%self.l
		return data

	# def has_next(self):
	# 	if self.read_ptr + 1 >= self.l:
	# 		self.read_ptr = 0
	# 	return self.buf[self.read_ptr + 1] is not None	



