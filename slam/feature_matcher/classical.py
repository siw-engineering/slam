import cv2
import numpy as np


class FrameMatcher(object):
	def __init__(self, matcher_type="bf"):
		self.ratio = 0.65
		if matcher_type == 'bf':
			self.matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
		elif matcher_type == 'flann':
			self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
	
	def filter_distance(self, matches):
		dist = [m.distance for m in matches]
		thres_dist = (sum(dist) / len(dist)) * self.ratio
		
		# keep only the reasonable matches
		pts1 = []
		pts2 = []
		for m in matches:
			if m.distance < thres_dist:
				pts1.append(self._kp1[m.queryIdx])
				pts2.append(self._kp2[m.trainIdx])
		return np.array(pts1), np.array(pts2)

	def match(self, kp1, d1, kp2, d2):
		self._kp1 = kp1
		self._kp2 = kp2
		matches = self.matcher.match(d1, d2)
		return self.filter_distance(matches)