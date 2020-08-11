import imgutils
import numpy as np
from slam.math.se3 import SE3_Exp

def residual_map(f1, f2, f1_d, K, xi, depth_scaling):

	residuals = np.zeros(f1.shape, dtype = np.float32)

	Kinv = np.linalg.inv(K)
	f = K[0,0]
	cx = K[0,2]
	cy = K[1,2]

	width = f1.shape[0]
	height = f1.shape[1]
	T = SE3_Exp(xi)
	for u in range(width):
		for v in range(height):
			Z = f1_d[u,v]/depth_scaling
			if Z <= 0:
				continue

			P = np.dot(Kinv, (u,v,Z))
			P = np.dot(T[0:3,0:3], np.asarray([X, Y, Z])) + T[0:3,3]
			P = np.reshape(P, (3,1))

			p_warped = np.dot(K, P)
			px = p_warped[0] / p_warped[2]
			py = p_warped[1] / p_warped[2]

			intensity_warped = imgutils.bilinear_interpolation(f2, px[0], py[0], width, height)

			if not np.isnan(intensity_warped):
				residuals.itemset((v, u), intensity_prev - intensity_warped)


	return residuals


def weighting(residuals, INITIAL_SIGMA, DEFAULT_DOF):
	w,h = residuals.shape
	weights = np.zeros(residuals.shape, dtype=np.float32)
	variance_init = 1.0 / (INITIAL_SIGMA * INITIAL_SIGMA)
	variance = variance_init
	num = 0.0
	dof = DEFAULT_DOF
	itr = 0
	while ((variance - variance_init) > 1e-3):
		itr += 1
		variance_init = variance
		variance = 0.0
		num = 0.0

		for i in range(w):
			for j in range(h):
				data = residuals[i, j]

				if not np.isnan(data):
					num += 1
					variance += data * data * ((dof + 1) /  (dof + variance_init * data * data))


	for i in range(w):
		for j in range(h):
			data = residuals[i, j]
			weights[i, j] = ( (dof + 1) / (dof + variance * data * data) )


	return weights