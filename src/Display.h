
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include "GPUTexture.h"
#include "shaders/Shaders.h"


class Display
{
public:
	Display(int w, int h)
	{
		width = w;
		height = h;
		widthPanel = 205;
		width += widthPanel;

		pangolin::Params windowParams;
		// windowParams.Set("SAMPLE_BUFFERS", 0);
		// windowParams.Set("SAMPLES", 0);
		pangolin::CreateWindowAndBind("Main", width, height, windowParams);
		// pangolin::CreateWindowAndBind("Main", width, height, pangolin::Params({{"scheme", "headless"}}));

		renderBuffer = new pangolin::GlRenderBuffer(1920, 1080);
		colorTexture = new GPUTexture(renderBuffer->width, renderBuffer->height, GL_RGBA32F, GL_RGBA, GL_FLOAT, true);

	    colorFrameBuffer = new pangolin::GlFramebuffer;
	    colorFrameBuffer->AttachColour(*colorTexture->texture);
	    colorFrameBuffer->AttachDepth(*renderBuffer);
	    test_program = std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface_phong.frag"));

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glDepthFunc(GL_LESS);

	}
	~Display()
	{}

	void draw()
	{
		colorFrameBuffer->Bind();
		glPushAttrib(GL_VIEWPORT_BIT);
		glViewport(0, 0, renderBuffer->width, renderBuffer->height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		colorFrameBuffer->Unbind();

	    glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFrameBuffer->fbid);

	    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	    glBlitFramebuffer(0, 0, renderBuffer->width, renderBuffer->height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

	    glBindTexture(GL_TEXTURE_2D, 0);

	    glFinish();
	}
	int width, height, widthPanel;
	pangolin::GlRenderBuffer* renderBuffer;
	pangolin::GlFramebuffer* colorFrameBuffer;
	GPUTexture* colorTexture;
	std::shared_ptr<Shader> test_program;
};