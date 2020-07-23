import cv2
import numpy as np

def ransac(p0, p1, Rt, K):
	R = Rt[:3, :3]
	t = Rt[:3, 3]
	focal = K[0, 0]
	pp = K[0, 3], K[1, 3]
	E, mask = cv2.findEssentialMat(p0,p1,focal,pp,cv2.FM_RANSAC, 0.999, 1.0)
	retval, R, t, mask = cv2.recoverPose(E, p0, p1, R, t, focal, pp, mask)
	Rt = np.eye(4)
	Rt[:3, :3] = R
	Rt[:3, 3] = t
	return Rt