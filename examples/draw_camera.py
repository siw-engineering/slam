import sys
sys.path.append("/home/developer/packages/pangolin/")
sys.path.append("/home/developer/slam/")

import numpy as np
import pangolin 
import OpenGL.GL as gl
from slam.visualizer.display import DisplayCam

cd = DisplayCam("cam")
cd.start()

while 1:
	pose = np.identity(4)
	pose[:3, 3] = np.random.randn(3)
	cd.q.put([pose])
