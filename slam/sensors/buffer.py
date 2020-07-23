import numpy as np
import sys

class Buffer:
	def __init__(self, l=100):
		self.l = l
		self.read_ptr = 0
		self.write_ptr = 0
		self.buf = [np.zeros(1)] * l 

	def add(self, data):

		if self.write_ptr >= self.l:
			self.write_ptr = 0

		self.buf[self.write_ptr] = data
		self.write_ptr += 1

	def read(self):
		if self.buf[self.read_ptr] is None:
			raise ValueError("Buffer element is None")

		if self.read_ptr >= self.l:
			print("read ptr 0")
			self.read_ptr = 0

		data = self.buf[self.read_ptr]
		self.read_ptr += 1
		return data

	def has_next(self):
		if self.read_ptr + 1 >= self.l:
			self.read_ptr = 0
		return self.buf[self.read_ptr + 1] is not None	


	def wait_for_fill(self):
		print("waiting for fill")
		while 1:
			e = 0
			for i in range(self.l):
				if self.buf[i].any():
					e += 1
			if e == self.l:
				print ("done")
				return
			pass
