import numpy as np
import sys

class Buffer:
	def __init__(self, l=100):
		self.l = l
		self.read_ptr = 0
		self.write_ptr = 0
		self.buf = [None] * self.l 

	def add(self, data):

		if self.write_ptr >= self.l:
			self.write_ptr = 0

		self.buf[self.write_ptr] = data
		self.write_ptr += 1

	def read(self):
		if self.buf[self.read_ptr] is None:
			raise ValueError("Buffer element is None")

		if self.read_ptr >= self.l:
			self.read_ptr = 0

		data = self.buf[self.read_ptr]
		self.read_ptr += 1
		return data

	def has_next(self):
		if self.read_ptr + 1 >= self.l:
			self.read_ptr = 0
			return False
		return self.buf[self.read_ptr + 1] is not None	