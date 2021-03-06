import cv2
import numpy as np

def by_ransac(p0, p1, K):
	focal = K[0, 0]
	pp = K[0, 2], K[1, 2]
	E, mask = cv2.findEssentialMat(p0,p1,focal,pp,cv2.FM_RANSAC, 0.999, 1.0)
	retval, R, t, mask = cv2.recoverPose(E, p0, p1)
	Rt = np.eye(4)
	Rt[:3, :3] = R
	Rt[:3, 3] = t.T
	return Rt