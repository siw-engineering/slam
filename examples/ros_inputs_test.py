import sys
sys.path.append("../")
from slam.sensors.camera import Camera
from slam.sensors.camera.inputs import RosTopicStream


rgb_cam = Camera(F=100, stream=RosTopicStream("/ROBOTIKA_X2/image_raw/compressed"))

while 1:
	if rgb_cam.has_next():
		img = rgb_cam.read()
		print (img.shape)
