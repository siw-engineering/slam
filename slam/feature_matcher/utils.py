import numpy as np

def filter_distance(matches, kp1, kp2, ratio):
	dist = [m.distance for m in matches]
	thres_dist = (sum(dist) / len(dist)) * ratio
	
	# keep only the reasonable matches
	pts1 = []
	pts2 = []
	for m in matches:
		if m.distance < thres_dist:
			pts1.append(kp1[m.queryIdx])
			pts2.append(kp2[m.trainIdx])
	return np.array(pts1), np.array(pts2)