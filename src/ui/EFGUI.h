#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <map>
#include "../sensors/Camera.h"

class EFGUI
{
public:
	EFGUI(float width, float height, CameraModel intr)
	{
		pangolin::OpenGlRenderState s_cam;
		pangolin::CreateWindowAndBind("Main",width, height);
		glEnable(GL_DEPTH_TEST);
		s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(width, height, intr.fx, intr.fy, intr.cx, intr.cy, 0.1, 1000),
		                                    pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));
		pangolin::View& d_cam = pangolin::CreateDisplay()
		     .SetBounds(0.0, 1.0, 0.0, 1.0, -width*height)
		     .SetHandler(new pangolin::Handler3D(s_cam));

	}
	~EFGUI()
	{
		
	}
	
};