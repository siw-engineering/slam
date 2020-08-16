import sys
sys.path.append("../")
sys.path.append("/home/developer/packages/pangolin/")
import cv2
import numpy as np
import pdb

from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import RosTopicStream
from slam.visualizer.display import Display2D
from slam.utils import residual_map


rgb_cam = Camera(cam_type="rgb8",F=100, stream=RosTopicStream("/ROBOTIKA_X2/image_raw"))
depth_cam = Camera(cam_type="32FC1",F=100, stream=RosTopicStream("/ROBOTIKA_X2/front/depth"))

res_display = Display2D("display")
res_display.start()
xi = np.ones(6)

while 1:
	f1= rgb_cam.read()
	f1_d = depth_cam.read()
	f2 = rgb_cam.read()
	res  = residual_map(cv2.resize(f1[:,:,0],(80,80)) , cv2.resize(f2[:,:,0], (80,80)), cv2.resize(f1_d, (80,80)), rgb_cam.K, xi)
	# res_display.q.put([res])
	print (res)