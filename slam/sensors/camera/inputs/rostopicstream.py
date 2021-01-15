import cv2
import rospy
import roslib
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import Process
import numpy as np
import time
import pdb

"""
reads ros topics
"""
class RosTopicStream(object):
	def __init__(self, topic):
		self.topic = topic

	def init_ros_node(self, cam_type):
		self.cam_type = cam_type
		if rospy.get_name() == "/unnamed":	
			# rospy.init_node('image_topic_reader_%s_%s'%(str(int(time.time())), cam_type), anonymous=True)
			rospy.init_node('image_topic_reader', anonymous=True)

	def read(self):
		ros_data = rospy.wait_for_message(self.topic, Image, timeout=None)
		image_np = CvBridge().imgmsg_to_cv2(ros_data, desired_encoding=self.cam_type)
		return image_np	
