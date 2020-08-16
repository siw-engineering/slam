import sys
sys.path.append("../")
sys.path.append("/home/developer/packages/pangolin/")

import cv2
import numpy as np
from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import VideoStream, RosTopicStream
from slam.visualizer.display import Display2D, DisplayCam, draw_matches
from slam.feature_extractor.classical import orb_features
from slam.feature_matcher.classical import FrameMatcher
import slam.pose_recovery as pose
import pdb

# rgb_cam = Camera(cam_type="rgb8",F=100, stream=VideoStream("/home/developer/dataset/car.mp4"))
rgb_cam = Camera(cam_type="rgb8",F=462.1, stream=RosTopicStream("/ROBOTIKA_X2/image_raw"))
rgb_cam.K = np.array(
					([462.1, 0.0, 320.5],
					[0.0, 462.1, 180.5],
					[0.0, 0.0, 1.0])
					)
# depth_cam = Camera(cam_type="32FC1",F=100, stream=RosTopicStream("/ROBOTIKA_X2/front/depth"))

rbg_display = Display2D("display")
depth_display = Display2D("display")
cam_display = DisplayCam("cam")

cam_display.start()
# rbg_display.start()
# depth_display.start()


rgb1 = rgb_cam.read()

feat = FrameMatcher("flann")
cam_pos = np.zeros((3,1)).T
Rf = None

while 1:
	rgb2 = rgb_cam.read()
	# dimg = depth_cam.read()
	f1 = orb_features(rgb1)
	f2 = orb_features(rgb2)
	f1['descriptors'] = np.float32(f1['descriptors'])
	f2['descriptors'] = np.float32(f2['descriptors'])

	pts1, pts2 = feat.match(f1, f2)
	if len(pts1) == 0 or len(pts2) == 0:
		print ("na")
		continue
	Rt = pose.by_ransac(pts1, pts2, rgb_cam.K)
	
	if Rf is None:
		Rf = Rt[:3,:3]

	rgb1 = rgb2
	cam_pos = cam_pos + np.dot(Rf, Rt[:3,3])
	Rf = np.dot(Rt[:3,:3],Rf)
	
	Rt[:3,3] = cam_pos
	cam_display.q.put([Rt])

	print (cam_pos)
	# rbg_display.q.put([draw_matches(pts1, pts2, rgb2)])



