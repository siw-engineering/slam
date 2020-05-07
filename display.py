from multiprocessing import Process, Queue
import numpy as np
import cv2
import OpenGL.GL as gl
import pangolin

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


'''
Class used for displaying 2D Data(frames from the queue)
'''

class Display2D(object):
	"""docstring for  Display2D"""
	def __init__(self, name):
		self.name = name
		self.q = Queue()
		self.p = Process(target=self.draw)
		self.p.daemon = True

	"""Display loop"""
	def draw(self):
		q = self.q
		while True:
			state = None
			try:
				while not q.empty():
					state = q.get() 
				if state is not None:
					cv2.imshow(self.name, state[0])
					cv2.waitKey(10)
			except:
				continue
	

	def start(self):
		self.p.start()

'''
Class used for displaying 3D Data
'''

class Display3D(object):
	"""docstring for  Display2D"""
	def __init__(self, name, history=False):
		self.process = Process(target=self.draw)
		self.process.daemon = True
		self.q = Queue()
		self.name = name
		self.history = history
		self.points = []

	def draw_init(self):
		pangolin.CreateWindowAndBind(self.name, 640, 480)
		gl.glEnable(gl.GL_DEPTH_TEST)
		self.scam = pangolin.OpenGlRenderState(
			pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
			pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
		handler = pangolin.Handler3D(self.scam)
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
		self.dcam.SetHandler(handler)



	def draw(self):
		q = self.q
		self.draw_init()
		while not pangolin.ShouldQuit():
			try:
				while not q.empty():
					state = q.get()

				if state is not None:

					gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
					gl.glClearColor(0.0, 0.0, 0.0, 0.0)
					self.dcam.Activate(self.scam)

					# Draw Point Cloud
					gl.glPointSize(2)
					gl.glColor3f(0.0, 1.0, 0.0)
					if self.history:
						self.points.append(state[0])
						pangolin.DrawPoints(np.reshape(np.array(self.points),(-1,3)))
					else:
						pangolin.DrawPoints(state[0])


					pangolin.FinishFrame()
			except:
				continue

	def start(self):
		self.process.start()


class DisplayFrustum(object):

	def __init__(self):
		
		print("DisplayFrustum")

	def draw_init(self, cam_loc):
		rect_1 = (cam_loc[0]*-1, cam_loc[1]*2, cam_loc[2]*2)
		rect_2 = (cam_loc[0]*-1, cam_loc[1]*2, cam_loc[2]*-2)
		rect_3 = (cam_loc[0]*-1, cam_loc[1]*-2, cam_loc[2]*-2)
		rect_4 = (cam_loc[0]*-1, cam_loc[1]*-2, cam_loc[2]*2)

		self.verticies = (
						rect_1,
						rect_2,
						rect_3,
						rect_4,
						cam_loc
				)
		self.edges = (
					(0, 1),
					(0, 3),
					(1, 2),
					(2, 3),
					(3, 0),
					(0, 4),
					(1, 4),
					(2, 4),
					(3, 4)
					)

	def Frustum(self):
		glBegin(GL_LINES)
		for edge in self.edges:
			for vertex in edge:
				glVertex3fv(self.verticies[vertex])
		glEnd()

	def draw(self, cam_loc):
		self.draw_init(cam_loc)
		pygame.init()
		display = (800,600)
		pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

		gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

		glTranslatef(0.0,0.0, -5)

		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					quit()

			glRotatef(1, 3, 1, 1)
			glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
			self.Frustum()
			pygame.display.flip()
			pygame.time.wait(10)

# df = DisplayFrustum()
# df.draw((-1, 0.5, 0.5))

def draw_matches(fm, t):
	img = fm._f1.img
	pts1 = t.toCameraSpace(fm.pts1)
	pts2 = t.toCameraSpace(fm.pts2)

	for pt1, pt2 in zip(pts1, pts2):
		pt1, pt2 = np.int32(pt1), np.int32(pt2)
		cv2.circle(img, tuple(pt1), color=(0,255,0), radius=3)
		cv2.circle(img, tuple(pt2), color=(0,255,0), radius=3)
		cv2.line(img, tuple(pt1), tuple(pt2), (0,0,255), 1)
	return img