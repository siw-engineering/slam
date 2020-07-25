import cv2
import numpy as np
from .utils import lows_filter


class FrameMatcher(object):
	def __init__(self, matcher="bf"):
		self.ratio = 0.65
		if matcher == 'bf':
			self.matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
		# not working
		elif matcher == 'flann':
			self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)


	def match(self, f1, f2):
		matches = self.matcher.match(f1['descriptors'], f2['descriptors'])
		return lows_filter(matches, f1['keypoints'], f2['keypoints'], self.ratio)