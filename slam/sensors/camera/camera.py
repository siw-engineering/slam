class Camera:
	def __init__(F=100, input_streamer=None):
		self.F = F
		self.input_streamer = input_streamer

	def getNext():
		pass

	def getPrev():
		pass
	
	@property
	def F(self):
		return self._F
	
	@property
	def K(self):
		return self._K

	@property
	def Kinv(self):
		return self._Kinv
	