# import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from .models.matching import Matching
from .models.utils import (AverageTimer,
							frame2tensor, get_pose)
import transformations as tf


class Odometry:
	
	def __init__(self, opt, frame):
		self.K = opt['K']
		self.isFirst = True
		self.traj = np.zeros([800, 800])
		# :/		
		self.config = {
							'superpoint': {
								'nms_radius': opt['nms_radius'],
								'keypoint_threshold': opt['keypoint_threshold'],
								'max_keypoints': opt['max_keypoints']
							},
							'superglue': {
								'weights': opt['superglue'],
								'sinkhorn_iterations': opt['sinkhorn_iterations'],
								'match_threshold': opt['match_threshold'],
							}
						}
		self.device = 'cuda' if torch.cuda.is_available() and not opt['force_cpu else'] else 'cpu'
		self.matching = Matching(self.config).eval().to(self.device)
		self.keys = ['keypoints', 'scores', 'descriptors']
		self.frame = frame
		self.frame_tensor = frame2tensor(self.frame, self.device)

		self.last_data = self.matching.superpoint({'image': self.frame_tensor})
		self.last_data = {k+'0': self.last_data[k] for k in self.keys}
		self.last_data['image0'] = self.frame_tensor
		self.last_frame = frame
		self.last_image_id = 0


	def get_poses(self, frame, i, t):

		# frame, ret = self.vs.next_frame()
		# timer.update('data')
		stem0, stem1 = self.last_image_id, i - 1

		frame_tensor = frame2tensor(frame, self.device)
		pred = self.matching({**self.last_data, 'image1': frame_tensor})
		kpts0 = self.last_data['keypoints0'][0].cpu().numpy()
		kpts1 = pred['keypoints1'][0].cpu().numpy()
		matches = pred['matches0'][0].cpu().numpy()
		confidence = pred['matching_scores0'][0].cpu().detach().numpy()
		# timer.update('forward')

		valid = matches > -1
		mkpts0 = kpts0[valid]
		mkpts1 = kpts1[matches[valid]]
		color = cm.jet(confidence[valid])

		thresh = 1
		# :/ opencv pose estimation
		# pose = get_pose(mkpts0, mkpts1, np.linalg.inv(self.K), thresh)
		# ransac sklearn
		E = tf.find_correspondace(mkpts0, mkpts1, t)
		pose = tf.recoverpose(E)
		self.last_data = {k+'0': pred[k+'1'] for k in self.keys}
		self.last_data['image0'] = frame_tensor
		self.last_frame = frame
		self.last_image_id = (i - 1)

		if pose is not None:
			'''
			self.rotationMat = rt[0]
			self.translationMat = rt[1]

			if (self.isFirst == True):

				self.translationMatrix = self.translationMat
				self.rotationMatrix = self.rotationMat
				self.isFirst = False

			# poses
			self.translationMatrix = self.translationMatrix + (self.rotationMatrix.dot(self.translationMat))
			self.rotationMatrix = self.rotationMatrix.dot(self.rotationMat)
			'''
		# return self.rotationMatrix, self.translationMatrix
			return pose
		else:
			return None


# if __name__ == "__main__":

# 	K = np.array([[574.54320988, 0., 322.7782716],
# 				  [0, 577.58181818, 238.80991736],
# 				  [0, 0, 1]])


# 	odometry = Odometry(K, "config.json")
# 	odometry.GetPoses()
