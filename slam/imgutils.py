import numpy as np
import cv2


	# Convert input color image to grayscale
def toGray(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def im2float(img):
	# Convert input image to normalized float
	return cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Function to downsample an intensity (grayscale) image
def downsampleGray(img):
	"""
	The downsampling strategy eventually chosen is a naive block averaging method.
	That is, for each pixel in the target image, we choose a block comprising 4 neighbors
	in the source image, and simply average their intensities. For each target image point 
	(y, x), where x indexes the width and y indexes the height dimensions, we consider the 
	following four neighbors: (2*y,2*x), (2*y+1,2*x), (2*y,2*x+1), (2*y+1,2*x+1).
	NOTE: The image must be float, to begin with.
	"""

	# Perform block-averaging
	img_new = (img[0::2,0::2] + img[0::2,1::2] + img[1::2,0::2] + img[1::2,1::2]) / 4.

	return img_new


# Function to downsample a depth image
def downsampleDepth(img):
	"""
	For depth images, the downsampling strategy is very similar to that for intensity images, 
	with a minor mod: we do not average all pixels; rather, we average only pixels with non-zero 
	depth values.
	"""

	# Perform block-averaging, but not across depth boundaries. (i.e., compute average only 
	# over non-zero elements)
	img_ = np.stack([img[0::2,0::2], img[0::2,1::2], img[1::2,0::2], img[1::2,1::2]], axis=2)
	num_nonzero = np.count_nonzero(img_, axis=2)
	num_nonzero[np.where(num_nonzero == 0)] = 1
	img_new = np.sum(img_, axis=2) / num_nonzero

	return img_new.astype(np.uint8)


def buildPyramid(f1, f1_d, f2, f2_d, K):
	pyramid_size = 4
	k_pyramid = []
	f1_pyramid = []
	f1_d_pyramid = []
	f2_pyramid = []
	f2_d_pyramid = []

	for level in range(pyramid_size):
		k_id = np.identity(3)
		k_id[:2, :3] = K[:2, :3]
		k_pyramid.append(k_id)
		f1_pyramid.append(f1)
		f1_d_pyramid.append(f1_d)
		f2_pyramid.append(f2)
		f2_d_pyramid.append(f2_d)
		# if level < pyramid_size -1:
		f1 =  downsampleGray(f1)
		f1_d = downsampleDepth(f1_d)
		f2 = downsampleGray(f2)
		f2_d = downsampleDepth(f2_d)
		K[:2, :3] = np.dot(k_id[:2, :3], 0.5)


	return f1_pyramid, f1_d_pyramid, f2_pyramid, f2_d_pyramid, k_pyramid