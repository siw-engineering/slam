import sys
sys.path.append("../")
# sys.path.append("/home/developer/packages/pangolin/")

import cv2
import numpy as np
from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import VideoStream
from slam.visualizer.display import Display2D, DisplayCam
from slam.feature_extractor.classical import orb_features
from slam.feature_matcher.classical import FrameMatcher
import slam.pose_recovery as pose

rgb_cam = Camera(cam_type="rgb8",F=100, stream=VideoStream("path/to/video.mp4"))
rbg_display = Display2D("display")
depth_display = Display2D("display")
cam_display = DisplayCam("cam")

cam_display.start()
rbg_display.start()
depth_display.start()


# depth_cam = Camera(cam_type="32FC1",F=100, stream=RosTopicStream("/ROBOTIKA_X2/front/depth"))
rgb1 = rgb_cam.read()

feat = FrameMatcher("bf")
cam_pos = np.zeros((3,1)).T
while 1:
	rgb2 = rgb_cam.read()
	# dimg = depth_cam.read()
	f1 = orb_features(rgb1)
	f2 = orb_features(rgb2)
	pts1, pts2 = feat.match(f1, f2)
	Rt = pose.by_ransac(pts1, pts2, rgb_cam.K)
	rgb1 = rgb2
	# cam_pos = cam_pos + np.subtract(Rt[:3,3], cam_pos)/rgb_cam.sf
	cam_display.q.put([Rt])
	# depth_display.q.put([dimg])

print ("out")

