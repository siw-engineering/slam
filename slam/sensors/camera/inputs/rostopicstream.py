import cv2
import rospy
import roslib
from sensor_msgs.msg import CompressedImage
from multiprocessing import Process
import numpy as np

# "/ROBOTIKA_X2/image_raw/compressed"
class RosTopicStream(object):
	def __init__(self, topic):
		self.buf = None
		rospy.init_node('image_topic_reader', anonymous=True)
		self.subscriber = rospy.Subscriber(topic, CompressedImage, self.callback,  queue_size = 1)

	def callback(self, ros_data):
		np_arr = np.fromstring(ros_data.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		if self.buf:
			self.buf.add(image_np)


	def spin(self):
		rospy.spin()

	def start(self, buf):
		self.buf = buf
		p = Process(target=self.spin, args=())
		p.start()
