import sys
sys.path.append("../")
sys.path.append("/home/developer/packages/pangolin/")

import cv2
from matplotlib import pyplot as plt
from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import RosTopicStream
from slam.visualizer.display import Display2D

rgb_cam = Camera(cam_type="rgb8",F=100, stream=RosTopicStream("/ROBOTIKA_X2/image_raw"))
# rbg_display = Display2D("rbg_display")
# rbg_display.start()
# depth_cam = Camera(cam_type="32FC1",F=100, stream=RosTopicStream("/ROBOTIKA_X2/front/depth"))

while rgb_cam.has_next():
	img = rgb_cam.read()
	plt.imshow("test", img)

	plt.show()
	# rbg_display.q.put([img])
	# print (rgb_cam._buf.read_ptr)
	# print (rgb_cam._buf.write_ptr)

print ("out")

