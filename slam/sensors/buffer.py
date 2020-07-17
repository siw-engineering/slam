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
			print("loop'd!")
			self.write_ptr = 0

		self.buf[self.write_ptr] = data
		print ("write!")
		self.write_ptr += 1


	def read(self):
		# print (self.buf[self.read_ptr])
		print (self.read_ptr)
		if self.buf[self.read_ptr] is None:
			raise ValueError("Buffer element is None")

		if self.read_ptr >= self.l:
			self.read_ptr = 0

		data = self.buf[self.read_ptr]
		self.read_ptr += 1
		return data

	def has_next(self):
		if self.read_ptr + 1 >= self.l:
			return False
		return self.buf[self.read_ptr + 1] is not None	