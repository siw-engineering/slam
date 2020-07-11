import cv2
import glob
import re
from pathlib import Path
import numpy as np
"""
Loads video files
"""
class VideoStreamer(object):
	def __init__(self, basedir, resize, skip, image_glob, max_length=1000000, preprocessor=None):

		self.preprocessor = preprocessor

		self._ip_grabbed = False
		self._ip_running = False
		self._ip_camera = False
		self._ip_image = None
		self._ip_index = 0
		self.cap = []
		self.camera = True
		self.video_file = False
		self.listing = []
		self.resize = resize
		self.interp = cv2.INTER_AREA
		self.i = 0
		self.skip = skip
		self.max_length = max_length
		if isinstance(basedir, int) or basedir.isdigit():
			print('==> Processing USB webcam input: {}'.format(basedir))
			self.cap = cv2.VideoCapture(int(basedir))
			self.listing = range(0, self.max_length)
		elif basedir.startswith(('http', 'rtsp')):
			print('==> Processing IP camera input: {}'.format(basedir))
			self.cap = cv2.VideoCapture(basedir)
			self.start_ip_camera_thread()
			self._ip_camera = True
			self.listing = range(0, self.max_length)
		elif Path(basedir).is_dir():
			print('==> Processing image directory input: {}'.format(basedir))
			self.listing = list(Path(basedir).glob(image_glob[0]))
			for j in range(1, len(image_glob)):
				image_path = list(Path(basedir).glob(image_glob[j]))
				self.listing = self.listing + image_path
			self.listing.sort()
			self.listing = self.listing[::self.skip]
			self.max_length = np.min([self.max_length, len(self.listing)])
			if self.max_length == 0:
				raise IOError('No images found (maybe bad \'image_glob\' ?)')
			self.listing = self.listing[:self.max_length]
			self.camera = False
		elif Path(basedir).exists():
			print('==> Processing video input: {}'.format(basedir))
			self.cap = cv2.VideoCapture(basedir)
			self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
			num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
			self.listing = range(0, num_frames)
			self.listing = self.listing[::self.skip]
			self.video_file = True
			self.max_length = np.min([self.max_length, len(self.listing)])
			self.listing = self.listing[:self.max_length]
		else:
			raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
		if self.camera and not self.cap.isOpened():
			raise IOError('Could not read camera')

	def __len__(self):
		return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	@property
	def w(self):
		return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	
	@property
	def h(self):
		return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	@property
	def frame(self):
		_, _frame = self.cap.read()
		if self.preprocessor:
			_frame = self.preprocessor(_frame)
		return _frame


	def load_image(self, impath):
		""" Read image as grayscale and resize to img_size.
		Inputs
			impath: Path to input image.
		Returns
			grayim: uint8 numpy array sized H x W.
		"""
		grayim = cv2.imread(impath, 0)
		if grayim is None:
			raise Exception('Error reading image %s' % impath)
		w, h = grayim.shape[1], grayim.shape[0]
		w_new, h_new = self.process_resize(w, h, self.resize)
		grayim = cv2.resize(
			grayim, (w_new, h_new), interpolation=self.interp)
		return grayim

	def next_frame(self):
		""" Return the next frame, and increment internal counter.
		Returns
			 image: Next H x W image.
			 status: True or False depending whether image was loaded.
		"""

		if self.i == self.max_length:
			return (None, False)
		if self.camera:

			if self._ip_camera:
				#Wait for first image, making sure we haven't exited
				while self._ip_grabbed is False and self._ip_exited is False:
					time.sleep(.001)

				ret, image = self._ip_grabbed, self._ip_image.copy()
				if ret is False:
					self._ip_running = False
			else:
				ret, image = self.cap.read()
			if ret is False:
				print('VideoStreamer: Cannot get image from camera')
				return (None, False)
			w, h = image.shape[1], image.shape[0]
			if self.video_file:
				self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

			w_new, h_new = self.process_resize(w, h, self.resize)
			image = cv2.resize(image, (w_new, h_new),
							   interpolation=self.interp)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		else:
			image_file = str(self.listing[self.i])
			image = self.load_image(image_file)
		self.i = self.i + 1
		return (image, True)

	def start_ip_camera_thread(self):
		self._ip_thread = Thread(target=self.update_ip_camera, args=())
		self._ip_running = True
		self._ip_thread.start()
		self._ip_exited = False
		return self

	def update_ip_camera(self):
		while self._ip_running:
			ret, img = self.cap.read()
			if ret is False:
				self._ip_running = False
				self._ip_exited = True
				self._ip_grabbed = False
				return

			self._ip_image = img
			self._ip_grabbed = ret
			self._ip_index += 1
			#print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


	def cleanup(self):
		self._ip_running = False

	# --- PREPROCESSING ---

	def process_resize(self, w, h, resize):
		assert(len(resize) > 0 and len(resize) <= 2)
		if len(resize) == 1 and resize[0] > -1:
			scale = resize[0] / max(h, w)
			w_new, h_new = int(round(w*scale)), int(round(h*scale))
		elif len(resize) == 1 and resize[0] == -1:
			w_new, h_new = w, h
		else:  # len(resize) == 2:
			w_new, h_new = resize[0], resize[1]

		# Issue warning if resolution is too small or too large.
		if max(w_new, h_new) < 160:
			print('Warning: input resolution is very small, results may vary')
		elif max(w_new, h_new) > 2000:
			print('Warning: input resolution is very large, results may vary')

		return w_new, h_new





"""
Loads images in path sorted based on idx in name
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