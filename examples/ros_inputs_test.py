import sys
sys.path.append("../")
sys.path.append("/home/developer/packages/pangolin/")

import cv2
from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import RosTopicStream
from slam.visualizer.display import Display2D

rgb_cam = Camera(cam_type="rgb8",F=100, stream=RosTopicStream("/ROBOTIKA_X2/image_raw"))
rbg_display = Display2D("display")
depth_display = Display2D("display")

rbg_display.start()
depth_display.start()

depth_cam = Camera(cam_type="32FC1",F=100, stream=RosTopicStream("/ROBOTIKA_X2/front/depth"))

while 1:
	img = rgb_cam.read()
	dimg = depth_cam.read()

	rbg_display.q.put([img])
	depth_display.q.put([dimg])

print ("out")

