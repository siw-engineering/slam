import numpy as np
import pdb

# def lows_filter(matches, kp1, kp2, ratio):

# 	pdb.set_trace()
# 	dist = [m.distance for m in matches]
# 	thres_dist = (sum(dist) / len(dist)) * ratio
	
# 	# keep only the reasonable matches
# 	pts1 = []
# 	pts2 = []
# 	for m in matches:
# 		if m.distance < thres_dist:
# 			pts1.append(kp1[m.queryIdx])
# 			pts2.append(kp2[m.trainIdx])
# 	return np.array(pts1), np.array(pts2)

def lowes_filter(matches, kp1, kp2):
	pts1 = []
	pts2 = []
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			pts1.append(kp1[m.queryIdx].pt)
			pts2.append(kp2[m.trainIdx].pt)

	return np.array(pts1), np.array(pts2)