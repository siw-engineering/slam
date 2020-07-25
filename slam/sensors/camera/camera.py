import numpy as np
from slam.sensors.camera.inputs import RosTopicStream

class Camera:
	def __init__(self, cam_type="rgb",F=None, Cx=100, Cy=100, sf=100, stream=None):
		self.F = F
		self.Cx = Cx
		self.Cy	= Cy
		#scaling factor
		self.sf = sf
		self.K =np.array([
			[F,0,Cx],
			[0,F,Cy],
			[0,0,1]
			]) 
		self.Kinv = np.linalg.inv(self.K)
		self.stream = stream
		if isinstance(stream, RosTopicStream):
			self.stream.init_ros_node(cam_type)

	def read(self):
		return self.stream.read()

	def has_next(self):
		return True
