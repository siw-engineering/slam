import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from .utils import lowes_filter


class FrameMatcher(object):
	def __init__(self, matcher="bf"):
		self.ratio = 0.65
		self.matcher = matcher


	def match(self, f1, f2):
		if self.matcher == 'bf':
			bf = cv2.BFMatcher(cv2.NORM_HAMMING)
			# Match descriptors.
			matches = bf.knnMatch(f1['descriptors'], f2['descriptors'],k=2)
		elif self.matcher == 'flann':
			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks=50)   # or pass empty dictionary
			flann = cv2.FlannBasedMatcher(index_params,search_params)
			matches = flann.knnMatch(f1['descriptors'], f2['descriptors'], k=2)

		pts1, pts2 = lowes_filter(matches, f1['keypoints'], f2['keypoints'])

		if len(pts1) == 0 or len(pts2) == 0:
			return [], []

		# try:
		# 	model, inliers = ransac((pts1,pts2),EssentialMatrixTransform,
		# 							min_samples=8, residual_threshold=.005,max_trials=100)
		# 	pts1 = pts1[inliers]
		# 	pts2 = pts2[inliers]
		# except Exception as e:
		# 	print ("Exception inliers")
		# 	return [], []

		return np.array(pts1), np.array(pts2)