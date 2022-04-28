#include <iostream>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>
#include "../gl/types.h"
#include <map>
#include "../gl/Shaders.h"
#include <Eigen/LU>

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
		std::shared_ptr<Shader> draw_model_program;
		std::shared_ptr<Shader> drawbbox_program;
		std::shared_ptr<Shader> drawcam_program;
		pangolin::Var<bool> *draw_boxes, *draw_mask, *draw_cam, *pause;

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

			int widthPanel = 200;
			pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(widthPanel));

		    pause = new pangolin::Var<bool>("ui.Pause", false, true);
			draw_cam = new pangolin::Var<bool>("ui.draw_cam", false, true);
			draw_boxes = new pangolin::Var<bool>("ui.draw_boxes", false, true);
			draw_mask = new pangolin::Var<bool>("ui.draw_mask", false, true);


			draw_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface_.vert","draw_global_surface_.frag", shader_dir));
			draw_model_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface_.vert","draw_global_surface_.frag", shader_dir));
			drawbbox_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_bbox.vert","draw_bbox.frag", shader_dir));
			drawcam_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_cam.vert", "draw_cam.frag", shader_dir));


		}

		pangolin::OpenGlMatrix getMVP()
		{
			pangolin::OpenGlMatrix view = s_cam.GetModelViewMatrix();
			pangolin::OpenGlMatrix projection = s_cam.GetProjectionMatrix();
			pangolin::OpenGlMatrix mvp =  projection * view;
			return mvp;
		}	
        void renderImg(GPUTexture * img)
        {

            glDisable(GL_DEPTH_TEST);

            d_img1.Activate();
            img->texture->RenderToViewport(true);

            glEnable(GL_DEPTH_TEST);
        }

        void renderMask(GPUTexture * img, GPUTexture * rawRgb)
        {

            glDisable(GL_DEPTH_TEST);

            d_img2.Activate();

            img->texture->RenderToViewport(true);

            glEnable(GL_DEPTH_TEST);
            pangolin::FinishFrame();

        }
		void renderModel(const std::pair<GLuint, GLuint>& vbos, int vs, bool maskDraw, GPUTexture * mask, Eigen::Matrix4f pose)
		{
			pangolin::Display("cam").Activate(s_cam);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glColor4f(1.0f,1.0f,1.0f,1.0f);

			Eigen::Matrix4f t_inv = pose.inverse();

			draw_program->Bind();
			draw_program->setUniform(Uniform("MVP", getMVP()));
			draw_program->setUniform(Uniform("maskDraw", maskDraw));
			draw_program->setUniform(Uniform("t_inv", t_inv));

			glBindBuffer(GL_ARRAY_BUFFER, vbos.first);

			glBindTexture(GL_TEXTURE_2D, mask->texture->tid);

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

		void renderModel(const std::pair<GLuint, GLuint>& vbos, int vs, Eigen::Matrix4f pose)
		{
			pangolin::Display("cam").Activate(s_cam);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glColor4f(1.0f,1.0f,1.0f,1.0f);

			Eigen::Matrix4f t_inv = pose.inverse();

			draw_model_program->Bind();
			draw_model_program->setUniform(Uniform("MVP", getMVP()));
			draw_model_program->setUniform(Uniform("t_inv", t_inv));

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

			draw_model_program->Unbind();
			pangolin::FinishFrame();
			
		}
		void renderCam(const Eigen::Matrix4f & pose, float width, float height, Eigen::Matrix3f & Kinv)
		{

		    drawcam_program->Bind();
		    drawcam_program->setUniform(Uniform("MVP", getMVP()));
		    drawcam_program->setUniform(Uniform("pose", pose));

		    glLineWidth(2);
		    pangolin::glDrawFrustum(Kinv, width, height, pose, 0.2f);
		    glLineWidth(1);
		    drawcam_program->Unbind();

		}

		void renderLiveBBox(GLfloat *& bbox_vertices_ptr, GLushort *& bbox_elements_ptr, int no, const Eigen::Matrix4f & pose)
		{

		    GLuint vbo_vertices;
		    glGenBuffers(1, &vbo_vertices);
		    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
		    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*8*4*no, bbox_vertices_ptr, GL_STATIC_DRAW);
		    glBindBuffer(GL_ARRAY_BUFFER, 0);

		    GLuint ibo_elements;
		    glGenBuffers(1, &ibo_elements);
		    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
		    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort)*24*no, bbox_elements_ptr, GL_STATIC_DRAW);
		    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		    drawbbox_program->Bind();
		    drawbbox_program->setUniform(Uniform("MVP", getMVP()));
		    drawbbox_program->setUniform(Uniform("pose", pose));

		    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
		    glEnableVertexAttribArray(0);
		    glVertexAttribPointer(0, 4,GL_FLOAT,GL_FALSE, 0,(void*)0);


		    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
		    glDrawElements(GL_LINES, 24*no, GL_UNSIGNED_SHORT, 0);
		    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		    glDisableVertexAttribArray(0);
		    glDisableVertexAttribArray(1);


		    glBindBuffer(GL_ARRAY_BUFFER, 0);
		    glDeleteBuffers(1, &vbo_vertices);
		    glDeleteBuffers(1, &ibo_elements);
		    drawbbox_program->Unbind();


		}


};