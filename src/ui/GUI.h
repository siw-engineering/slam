#include <iostream>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>
#include "../gl/types.h"
#include <map>
#include "../gl/Shaders.h"

class GUI
{
	public:

		pangolin::OpenGlRenderState s_cam;
		unsigned char* imageArray;
		pangolin::View d_img1;
		pangolin::View d_img2;
		int width = 1280;
		int height = 980;
		int width_img;
		int height_img;

		std::shared_ptr<Shader> draw_program;
		pangolin::OpenGlMatrix mvp;

		GUI(std::string shader_dir)
		{


			pangolin::Params windowParams;

			windowParams.Set("SAMPLE_BUFFERS", 0);
			windowParams.Set("SAMPLES", 0);

			// create OPENGL window in a single line
			pangolin::CreateWindowAndBind("Main",width, height, windowParams);

			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);

            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);

            s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 277, 277, 640 / 2.0f, 480 / 2.0f, 0.1, 1000),
                                                pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));

            pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

			d_img1 = pangolin::Display("img1")
				.SetAspect(640.0f/480.0f);

			d_img2 = pangolin::Display("img2")
				.SetAspect(640.0f/480.0f);

			pangolin::Display("multi")
				.SetBounds(pangolin::Attach::Pix(0), 1 / 4.0f, pangolin::Attach::Pix(180), 1.0)
				.SetLayout(pangolin::LayoutEqualHorizontal)
				.AddDisplay(d_img1)
				.AddDisplay(d_img2);

			draw_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface_.vert","draw_global_surface_.frag", shader_dir));
		}

		pangolin::OpenGlMatrix getMVP()
		{
			pangolin::OpenGlMatrix view = s_cam.GetModelViewMatrix();
			pangolin::OpenGlMatrix projection = s_cam.GetProjectionMatrix();
			pangolin::OpenGlMatrix mvp =  projection * view;
			return mvp;
		}	
        void displayImg(GPUTexture * img)
        {

            glDisable(GL_DEPTH_TEST);

            d_img1.Activate();
            img->texture->RenderToViewport(true);

            glEnable(GL_DEPTH_TEST);
        }

        void drawMask(GPUTexture * img, GPUTexture * rawRgb)
        {

            glDisable(GL_DEPTH_TEST);

            d_img2.Activate();

            img->texture->RenderToViewport(true);

            glEnable(GL_DEPTH_TEST);
            pangolin::FinishFrame();

        }
		void drawModel(const std::pair<GLuint, GLuint>& vbos, int vs)
		{
			pangolin::Display("cam").Activate(s_cam);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			draw_program->Bind();
			draw_program->setUniform(Uniform("MVP", getMVP()));

			glBindBuffer(GL_ARRAY_BUFFER, vbos.first);

			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vs, 0);

			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vs, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vs, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glDrawTransformFeedback(GL_POINTS, vbos.second);  // RUN GPU-PASS

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			draw_program->Unbind();
#if !enableMultipleModel			
			pangolin::FinishFrame();
#endif // multiplemodel
		}
};