import cv2
import numpy as np


def orb_features(img):
	orb = cv2.ORB_create()
	pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

	# extraction
	kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
	pts, des = orb.compute(img, kp)

	return pts, des

def sift_features(img):
	sift = cv2.xfeatures2d.SIFT_create()
	pts, des = sift.detectAndCompute(img, None)

	return pts, des
