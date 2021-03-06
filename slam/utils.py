import numpy as np
from slam.math.se3 import SE3_Exp, SO3_hat, SO3_left_jacobian, SE3_left_jacobian
from slam.math import lie

import pdb

# Interpolate an intensity value bilinearly (useful when a warped point is not integral)
# Translated from: https://github.com/muskie82/simple_dvo/blob/master/src/util.cpp
def bilinear_interpolation(img, x, y, width, height):

	# Consider the pixel as invalid, to begin with
	valid = np.nan

	# Get the four corner coordinates for the current floating point values x, y
	
	x0 = np.floor(x)
	y0 = np.floor(y)

	x0 = x0.astype(np.uint64)
	y0 = y0.astype(np.uint64)
	if type(x0) == np.ndarray:
		x0 = x0[0]
	if type(y0) == np.ndarray:
		y0 = y0[0]

	x1 = x0 + 1
	y1 = y0 + 1

	x1 = x1.astype(np.uint64)
	y1 = y1.astype(np.uint64)

	# Compute weights for each corner location, inversely proportional to the distance
	x1_weight = x - x0
	y1_weight = y - y0
	x0_weight = 1 - x1_weight
	y0_weight = 1 - y1_weight

	# Check if the warped points lie within the image
	if x0 < 0 or x0 >= width:
		x0_weight = 0
	if x1 < 0 or x1 >= width:
		x1_weight = 0
	if y0 < 0 or y0 >= height:
		y0_weight = 0
	if y1 < 0 or y1 >= height:
		y1_weight = 0

	# Bilinear weights
	w00 = x0_weight * y0_weight
	w10 = x1_weight * y0_weight
	w01 = x0_weight * y1_weight
	w11 = x1_weight * y1_weight

	# Bilinearly interpolate intensities

	sum_weights = w00 + w10 + w01 + w11
	total = 0
	total = np.int64(total)
	if w00 > 0:
		total += img.item((y0, x0)) * w00
	if w01 > 0:
		total += img.item((y1, x0)) * w01
	if w10 > 0:
		total += img.item((y0, x1)) * w10
	if w11 > 0:
		total += img.item((y1, x1)) * w11

	if sum_weights > 0:
		valid = total / sum_weights

	return valid


def residual_map(f1, f2, f1_d, K, xi, depth_scaling=1):

	residuals = np.zeros(f1.shape, dtype = np.float32)

	Kinv = np.linalg.inv(K)
	f = K[0,0]
	cx = K[0,2]
	cy = K[1,2]

	width = f1.shape[1]
	height = f1.shape[0]
	Rt = lie.se3(vector=xi).exp().matrix()
	for v in range(height):
		for u in range(width):
			intensity_prev = f1.item((v,u))
			Z = f1_d[v, u]
			if Z <= 0:
				continue

			P = np.dot(Kinv, (v, u, Z))
			P = np.dot(Rt[0:3,0:3], P) + Rt[0:3,3]
			P = np.reshape(P, (3,1))

			# print ("%f, %f\n"%(px,py))
			if P[2,0] > 0:
				p_warped = np.dot(K, P)
				px = p_warped[0] / p_warped[2]
				py = p_warped[1] / p_warped[2]
				intensity_warped = bilinear_interpolation(f2, px[0], py[0], width, height)

				if not np.isnan(intensity_warped):
					residuals.itemset((v, u), intensity_prev - intensity_warped)
				

	return residuals


def dvo_weighting(residuals, INITIAL_SIGMA=5, DEFAULT_DOF=5):
	w,h = residuals.shape
	weights = np.zeros(residuals.shape, dtype=np.float32)
	lambda_init = 1.0 / (INITIAL_SIGMA * INITIAL_SIGMA)
	lambda_ = lambda_init
	num = 0.0
	dof = DEFAULT_DOF
	itr = 0
	while True:
		itr += 1
		lambda_init = lambda_
		lambda_ = 0.0
		num = 0.0

		for i in range(w):
			for j in range(h):
				data = residuals[i, j]

				if not np.isnan(data):
					num += 1
					lambda_ += data * data * ((dof + 1) /  (dof + lambda_init * data * data))
		lambda_ /= num
		lambda_ = 1.0 / lambda_
		if not (abs(lambda_ - lambda_init) > 1e-3):
			break

	for i in range(w):
		for j in range(h):
			data = residuals[i, j]
			weights[i, j] = ( (dof + 1) / (dof + lambda_ * data * data) )

	return weights

# Function to compute image gradients (used in Jacobian computation)
def computeImageGradients(img):
	"""
	We use a simple form for the image gradient. For instance, a gradient along the X-direction 
	at location (y, x) is computed as I(y, x+1) - I(y, x-1).
	"""
	gradX = np.zeros(img.shape, dtype = np.float32)
	gradY = np.zeros(img.shape, dtype = np.float32)

	width = img.shape[1]
	height = img.shape[0]

	# Exploit the fact that we can perform matrix operations on images, to compute gradients quicker
	gradX[:, 1:width-1] = (img[:, 2:] - img[:,0:width-2]) * 0.5
	gradY[1:height-1, :] = (img[2:, :] - img[:height-2, :]) * 0.5

	return gradX, gradY

def computeJacobian(f1, f1_d, f2, K, xi, residuals, depth_scaling):
	width = f1.shape[0]
	height = f1.shape[1]
	Kinv = np.linalg.inv(K)
	grad_ix, grad_iy = computeImageGradients(f1)

	f = K[0, 0]
	cx = K[0, 2]
	cy = K[1, 2]
	Rt = lie.se3(vector=xi).exp().matrix()
	J = np.zeros([height*width, 6])


	for u in range(f1.shape[0]):
		for v in range(f1.shape[1]):
			Z = f1_d[u, v]/ depth_scaling
			if Z <= 0:
				continue
			P = np.dot(Kinv, (u,v,Z))
			P = np.dot(Rt[0:3,0:3], P) + Rt[0:3,3]
			P = np.reshape(P, (3,1))

			J_img = np.reshape(np.asarray([[grad_ix[u,v], grad_iy[u,v]]]), (1,2))
			J_pi = np.reshape(np.asarray([[f/P[2], 0, -f*P[0]/(P[2]*P[2])], [0, f/2, -f*P[1]/(P[2]*P[2])]]), (2,3))
			J_exp = np.concatenate((np.eye(3), SO3_hat(-P)), axis=1)

			J_exp = np.dot(J_exp, SE3_left_jacobian(xi))
			J[u*width+v,:] = residuals[u,v] * np.reshape(np.dot(J_img, np.dot(J_pi, J_exp)), (6))
	return J

def jacobian(f1, f1_d, f2, K, Kinv, xi, residuals):

	grad_ix, grad_iy = computeImageGradients(f1)
	J1 = np.zeros([1,2])
	h,w = f1.shape
	J = np.zeros([h*w, 6])
	fx = K[0,0]
	fy = K[1,1]
	cx = K[0,2]
	cy = K[1,2]
	fxi = 1/fx
	fyi = 1/fy

	Rt = lie.se3(vector=xi).exp().matrix()
	for v in range(h):
		for u in range(w):
			Z = f1_d[v, u]
			if Z <= 0:
				continue
			P = np.dot(Kinv, (v, u, Z))
			P = np.dot(Rt[0:3,0:3], P) + Rt[0:3,3]
			P = np.reshape(P, (3,1))
			Jw = np.array([fx*1/P.T[0,2], 0, -fx*P.T[0,0]/(P.T[0,2]*P.T[0,2]), -fx*(P.T[0,0]*P.T[0,1])/(P.T[0,2]*P.T[0,2]), fx*(1 + (P.T[0,0]*P.T[0,0])/(P.T[0,2]*P.T[0,2])), -fx*P.T[0,1]/P.T[0,2],0, fy*1/P.T[0,2], -fy*P.T[0,1]/(P.T[0,2]*P.T[0,2]), -fy*(1+(P.T[0,1]*P.T[0,1])/(P.T[0,2]*P.T[0,2])), fy*P.T[0,0]*P.T[0,1]/(P.T[0,2]*P.T[0,2]), fy*P.T[0,0]/P.T[0,2]]).reshape(2,6)

		if P[2,0] > 0:
			p2_warped = np.dot(K, P)
			px = p2_warped[0] / p2_warped[2];
			py = p2_warped[1] / p2_warped[2];
			J1[0,0] = bilinear_interpolation(grad_ix, px, py, w, h);
			J1[0,1] = bilinear_interpolation(grad_iy, px ,py, w, h);

			J[v*h+u,:] = -np.dot(J1,Jw)

		if np.isnan(J[v*h+u,0]):
			J[v*h+u,:] = np.zeros([1,6])

	return J