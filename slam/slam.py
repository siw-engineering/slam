#!/usr/bin/python3.7
import os
import cv2
import numpy as np
import argparse
import data_loader
from display import Display2D, Display3D, draw_matches
from frame import Frame, FrameMatcher
from transformations import Transform, recoverpose


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="/home/christie/projects/work/siw/slam/driving-videos/2.mp4")

args = parser.parse_args()
data_path = args.data_path

"""
check if data is Video or Images
"""
if os.path.isfile(data_path):
	data = data_loader.Video(path=data_path, preprocessor=lambda img: cv2.resize(img,(img.shape[1]//2, img.shape[0]//2)))
else:
	data = data_loader.Images(path=data_path, pattern="*_rgb.jpg", preprocessor=lambda img: cv2.resize(img,(img.shape[1]//2, img.shape[0]//2)))


#create displays
rgb_display = Display2D(name="rbg")
cam_pose_display = Display3D(name="Odometry", history=True)

#start displays
rgb_display.start()
cam_pose_display.start()


#Defines camera intrinsics and transformation b/w image and world space
t = Transform(F=100, W=data.w//2, H=data.h//2)

'''
f1 - previous frame
f2 - current frame
'''
#init cam position
cam_pos = np.zeros((3,1)).T

f1 = Frame(data.frame)
for i in range(len(data)):
	f2 = Frame(data.frame)
	fm = FrameMatcher(f1, f2, t)
	#recover pose from Essential Matrix
	f2.pose = recoverpose(fm.E)

	cam_pos = cam_pos + np.subtract(f2.pose[:3,3], cam_pos)/100
	cam_pose_display.q.put([cam_pos])
	rgb_display.q.put([draw_matches(fm, t)])
	f1 = f2