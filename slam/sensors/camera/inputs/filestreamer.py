import cv2
import glob
import r

"""
Loads images in path sorted based on idx in name
it assumes all images to have the same widht and height and 
contained in a single dir
"""
class Images(object):
	def __init__(self, path, pattern="*", preprocessor=None):
		self.path = path
		files = glob.glob("%s/%s"%(self.path,pattern))
		files.sort(key=lambda f: int(re.sub('\D', '', f)))
		self.files = files
		self.ptr = 0
		self.preprocessor = preprocessor

	def __len__(self):
		return len(self.files)

	@property
	def frame(self):
		if self.ptr > len(self.files):
			return None
		else:
			self.ptr +=1
			_frame = cv2.imread(self.files[self.ptr-1])	
			if self.preprocessor:
				_frame  = self.preprocessor(_frame)
				return _frame