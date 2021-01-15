import cv2
import numpy as np

def sift_features(img):
	feature = cv2.xfeatures2d.SIFT_create()
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	feature_pts = [x.pt for x in feature_pts]
	return {
		'keypoints': feature_pts,
		'descriptors': descriptors,
	}


# def orb_features(img):
# 	feature = cv2.ORB_create(nfeatures=1500)
# 	feature_pts, descriptors = feature.detectAndCompute(img, None)
# 	scores = 0
# 	# feature_pts = [x.pt for x in feature_pts]
# 	return {
# 		'keypoints': feature_pts,
# 		'descriptors': descriptors,
# 	}

def orb_features(img):
	orb = cv2.ORB_create()
	pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

	# extraction
	kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
	pts, des = orb.compute(img, kp)

	return {
		'keypoints': pts,
		'descriptors': des,
	}


def surf_features(img):
	feature = cv2.xfeatures2d.SURF_create()
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	feature_pts = [x.pt for x in feature_pts]
	return {
		'keypoints': feature_pts,
		'descriptors': descriptors,
	}


def good_features(img):
	pass


def fast_feature(img):
	feature = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
	feature_pts, descriptors = feature.detectAndCompute(img, None)
	scores = 0
	feature_pts = [x.pt for x in feature_pts]
	return {
		'keypoints': feature_pts,
		'descriptors': descriptors,
	}