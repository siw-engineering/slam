import numpy as np

class Transform:
	def __init__(self, K):
		self.K =np.array(K)
		self.Kinv = np.linalg.inv(self.K)

	def toWorldSpace(self, pts):
		pts_new = []
		for pt in pts:
			pts_new.append(np.dot(self.Kinv, np.append(pt,1).T).T)
		return np.array(pts_new)[:,0:2]


	def toCameraSpace(self, pts):
		pts_new = []
		for pt in pts:
			pts_new.append(np.dot(self.K, np.append(pt,1).T).T)
		return np.array(pts_new)[:,0:2]


def poseRt(R, t):
	ret = np.eye(4)
	ret[:3, :3] = R
	ret[:3, 3] = t
	return ret

def recoverpose(E):
	W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
	U,d,VT= np.linalg.svd(E)

	assert np.linalg.det(U)> 0
	if np.linalg.det(VT) < 0:
		VT *= -1

	R = np.dot(np.dot(U,W),VT)
	if np.sum(R.diagonal()) < 0:
		R = np.dot(np.dot(U,W.T),VT)

	t = U[:,2]
	#make Rt
	Rt = poseRt(R,t)
	return Rt

