#!/usr/bin/python3.7
import os
import json
import cv2
import numpy as np
from data_loader import VideoStreamer
# from display import Display2D, Display3D, draw_matches
from frame import Frame, FrameMatcher
from transformations import Transform, recoverpose
from feature_extractor.super_glue import Odometry


config = json.load(open("params/config.json", "r"))['SuperGlue']

# load data from different source
vs = VideoStreamer(config['input'], config['resize'], config['skip'],
				   config['image_glob'], config['max_length'])


'''
#create displays
rgb_display = Display2D(name="rbg")
cam_pose_display = Display3D(name="Odometry", history=True)

#start displays
rgb_display.start()
cam_pose_display.start()
'''

#Defines camera intrinsics and transformation b/w image and world space
t = Transform(config['K'])

'''
f1 - previous frame
f2 - current frame
'''
#init cam position
# first frame :/
frame, ret = vs.next_frame()

odom = Odometry(config, frame)

while True:
	frame, ret = vs.next_frame()
	# f2 = Frame(frame)
	# fm = FrameMatcher(f1, f2, t)
	# f2.pose = recoverpose(fm.E)

	if not ret:
		break
	# pose : 4x4 R&T mat
	pose = odom.get_poses(frame, vs.i, t)
	print(pose)

	# cam_pos = cam_pos + np.subtract(f2.pose[:3,3], cam_pos)/100
	# cam_pose_display.q.put([cam_pos])
	# rgb_display.q.put([draw_matches(fm, t)])

	# f1 = f2
