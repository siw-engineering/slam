from slam.sensors import Buffer
import numpy as np

class Camera:
	_buf = None
	def __init__(self, cam_type="rgb",F=None, Cx=100, Cy=100, stream=None, l=100):
		self.F = F
		self.Cx = Cx
		self.Cy	= Cy
		self.K =np.array([
			[F,0,Cx],
			[0,F,Cy],
			[0,0,1]
			]) 
		self.Kinv = np.linalg.inv(self.K)
		self.stream = stream
		self.stream.init_ros_node(cam_type)
		self._buf = Buffer(l=l)
		self.stream.start(self._buf)

	def read(self):
		data = self._buf.read()
		return data

	def has_next(self):
		return self._buf.has_next()
