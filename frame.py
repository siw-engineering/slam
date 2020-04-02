import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
import feature_extractor as fe


class Frame(object):
	def __init__(self, img):
		self.pose = np.identity(4)
		self._img = img

	
	@property
	def img(self):
		return self._img


	
"""
Class responsible for frame matching
"""
class FrameMatcher(object):
	def __init__(self, f1, f2, t, matcher="bf", featext='orb'):
		self._f1 = f1
		self._f2 = f2
		self.E = None
		if featext == 'orb':
			self._kp1, self._des1 = fe.orb_features(f1.img)
			self._kp2, self._des2 = fe.orb_features(f2.img)
		elif featext == 'sift':
			self._kp1, self._des1 = fe.sift_features(f1.img)
			self._kp2, self._des2 = fe.sift_features(f2.img)

		if matcher == 'bf':
			matches = self.bf_matcher()
		elif matcher == 'flann':
			matches = self.flann_matcher()

		pts1, pts2 = self.lowes_filter(matches)


		pts1 = t.toWorldSpace(pts1)
		pts2 = t.toWorldSpace(pts2)


		model, inliers = ransac((pts1,pts2),EssentialMatrixTransform,
								min_samples=8, residual_threshold=.005,max_trials=100)
		pts1 = pts1[inliers]
		pts2 = pts2[inliers]
		
		self.E = model.params
		self._pts1 = np.array(pts1)
		self._pts2 = np.array(pts2)

		

	def flann_matcher(self):
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(self._des1, self._des2, k=2)
		return matches

	def bf_matcher(self):
		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		# Match descriptors.
		matches = bf.knnMatch(self._des1,self._des2,k=2)

		return matches

	def lowes_filter(self, matches):
		pts1 = []
		pts2 = []
		# ratio test as per Lowe's paper
		for i,(m,n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				pts1.append(self._kp1[m.queryIdx].pt)
				pts2.append(self._kp2[m.trainIdx].pt)

		return np.array(pts1), np.array(pts2)

	@property
	def pts1(self):
		return self._pts1
	
	@property
	def pts2(self):
		return self._pts2
	
	@property
	def f1(self):
		return self._f1

	@property
	def f2(self):
		return self._f2
	
	