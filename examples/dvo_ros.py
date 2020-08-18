import sys
sys.path.append("../")
sys.path.append("/home/developer/packages/pangolin/")
import cv2
import numpy as np

from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import RosTopicStream
from slam.visualizer.display import Display2D
from slam.utils import residual_map, dvo_weighting, computeJacobian


rgb_cam = Camera(cam_type="rgb8",F=100, stream=RosTopicStream("/ROBOTIKA_X2/image_raw"))
rgb_cam.K = np.array(
					([462.1, 0.0, 320.5],
					[0.0, 462.1, 180.5],
					[0.0, 0.0, 1.0])
					)
depth_cam = Camera(cam_type="32FC1",F=100, stream=RosTopicStream("/ROBOTIKA_X2/front/depth"))
depth_cam.K = np.array(
					([462.1, 0.0, 320.5],
					[0.0, 462.1, 180.5],
					[0.0, 0.0, 1.0])
					)


res_display = Display2D("display")
res_display.start()
xi = np.zeros((6,1))

# f1= cv2.resize(rgb_cam.read()[:,:,0],(80,80))
f1= rgb_cam.read()


#gauss newton params
gn_iteration = 100
width = 80
height = 80
while 1:
	f1_d = depth_cam.read()
	f2 = rgb_cam.read()
	res  = residual_map(cv2.resize(f1[:,:,0],(width,height)) , cv2.resize(f2[:,:,0], (width,height)), cv2.resize(f1_d, (width,height)), rgb_cam.K, xi)
	weights = dvo_weighting(res)
	res = weights* res
	J = computeJacobian(cv2.resize(f1[:,:,0],(width,height)), cv2.resize(f1_d, (width,height)), cv2.resize(f2[:,:,0], (width,height)), rgb_cam.K, xi, res, depth_scaling=1)
	for i in range(width):
		for j in range(height):
			for k in range(J.shape[1]):
				J[i*width+j,k] = J[i*width+j,k] * weights[i,j]

	error = res * res.T
	b = J.T * res.reshape(width*height)
	H = np.dot(J.T, J)
	# res_display.q.put([weights])
	# print (res)
	f1 = f2

