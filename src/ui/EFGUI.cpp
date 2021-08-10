#include "EFGUI.h"


EFGUI::EFGUI(float width, float height, float cx, float cy, float fx, float fy)
{
    pangolin::CreateWindowAndBind("Main",width, height);
    glEnable(GL_DEPTH_TEST);
    s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(width, height, 277, 277, width / 2.0f, height / 2.0f, 0.1, 1000),
                                        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));
    pangolin::View& d_cam = pangolin::CreateDisplay()
         .SetBounds(0.0, 1.0, 0.0, 1.0, -2*width*height)
         .SetHandler(new pangolin::Handler3D(s_cam));

    draw_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface_.vert","draw_global_surface_.frag", "/home/developer/slam/src/ui/shaders/"));

}

pangolin::OpenGlMatrix EFGUI::getMVP()
{
    pangolin::OpenGlMatrix view = s_cam.GetModelViewMatrix();
    pangolin::OpenGlMatrix projection = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix mvp =  projection * view;
    return mvp;
}	

void EFGUI::render(const std::pair<GLuint, GLuint>& vbos, int vs)
{
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

    pangolin::FinishFrame();

}