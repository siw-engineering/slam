
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <map>
#include "../gl/Shaders.h"

#ifndef EFGUI_H
#define EFGUI_H

class EFGUI
{
	public:
		EFGUI(float width, float height, float cx, float cy, float fx, float fy);
		// ~EFGUI();
		pangolin::OpenGlMatrix getMVP();
		void render(const std::pair<GLuint, GLuint>& vbos, int vs);
		pangolin::OpenGlRenderState s_cam;
		std::shared_ptr<Shader> draw_program;
};

#endif //EFGUI_H