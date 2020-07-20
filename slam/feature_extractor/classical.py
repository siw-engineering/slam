import cv2
import numpy as np


def sift(img, enable_score=False):
	feature = cv2.xfeatures2d.SIFT_create()
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	# feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)
	if enable_score:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}
	else:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}


def orb(img, enable_score=False):
	feature = cv2.ORB_create(nfeatures=1500)
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	# feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)
	if enable_score:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}
	else:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}


def surf(img, enable_score=False):
	feature = cv2.xfeatures2d.SURF_create()
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	# feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)
	if enable_score:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}
	else:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}


def good_features(img, enable_score=False):
	pass


def fast_feature(img, enable_score=False):
	feature = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	# feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)
	if enable_score:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}
	else:
		return {
			'keypoints': feature_pts,
			'scores': scores,
			'descriptors': descriptors,
		}