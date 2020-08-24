import sys
sys.path.append("../")
sys.path.append("/home/developer/packages/pangolin/")
import cv2
import numpy as np
from scipy import linalg
import sys
import pdb

from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import RosTopicStream
from slam.visualizer.display import Display2D
from slam.utils import residual_map, dvo_weighting, computeJacobian, jacobian
from slam.imgutils import buildPyramid
# from slam.math.se3 import SE3_Exp, SE3_Log
from slam.math import lie

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

f1= rgb_cam.read()


#gauss newton params
gn_iterations = 100
width = 80
height = 80



while 1:
	Rt = np.identity(4)
	f1_d = depth_cam.read()
	f2 = rgb_cam.read()
	# f1 = cv2.resize(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY),(width,height)).astype(np.float32)
	f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
	f1 /= 255

	# f2 = cv2.resize(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), (width,height)).astype(np.float32)
	f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
	f2 /= 255

	# f1_d = cv2.resize(f1_d, (width,height)).astype(np.float32)
	f1_d = f1_d.astype(np.float32)

	# f1_pyr, f1_d_pyr, f2_pyr, k_pyr = buildPyramid(f1, f1_d, f2, rgb_cam.K)

	for level in range(3):
		error_prev = sys.float_info.max
		# f1 = f1_pyr[2]
		# f1_d = f1_d_pyr[2]
		# f2 = f2_pyr[2]
		width = f1.shape[1]
		height = f1.shape[0]
		xi = lie.SE3(Rt).log().vector()
		for itr in range(gn_iterations):
			res  = residual_map(f1, f2, f1_d, rgb_cam.K, xi)
			pdb.set_trace()
			weights = dvo_weighting(res)
			res = weights * res
			J = jacobian(f1, f1_d, f2, rgb_cam.K, rgb_cam.Kinv, xi, res)
			# J1 = computeJacobian(f1, f1_d, f2, rgb_cam.K, xi, res, depth_scaling=1)

			for i in range(height):
				for j in range(width):
					for k in range(J.shape[1]):
						J[i*width+j,k] = J[i*width+j,k] * weights[i,j]
			# error = res * res.T
			b = np.dot(J.T,res.reshape((width*height,1)))
			H = np.dot(J.T, J)
			print (H)
			print ("itr :%d\n"%itr)
			inc = -linalg.solve(linalg.cholesky(H),b)
			# zz = lie.SE3(lie.se3(xi).exp() * lie.se3(inc).exp()).log().vector()
			zz = lie.SE3((lie.se3(vector=xi).exp() * lie.se3(vector=xi).exp()).matrix()).log().vector()
			# res_display.q.put([weights])
			print (zz)
			f1 = f2

