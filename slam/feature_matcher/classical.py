import cv2
import numpy as np
from .utils import lows_filter


class FrameMatcher(object):
	def __init__(self, matcher="bf"):
		self.ratio = 0.65
		if matcher == 'bf':
			self.matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
		elif matcher == 'flann':
			self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)


	def match(self, kp1, d1, kp2, d2):
		matches = self.matcher.match(d1, d2)
		return lows_filter(matches, kp1, kp2, self.ratio)