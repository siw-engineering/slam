#include <pangolin/pangolin.h>
#include <iostream>
#include "Camera.h"
#include "RGBDOdometry.h"
#include "ComputePack.h"
#include "FeedbackBuffer.h"
#include "LogReader.h"
#include "RawLogReader.h"
#include "Vertex.h"
#include "GlobalModel.h"
#include "FillIn.h"
#include "Ferns.h"
#include "Resolution.h"
#include "Intrinsics.h"
#include "Deformation.h"
#include "PoseMatch.h"
#include "Resize.h"

void predict(IndexMap& indexMap, Eigen::Matrix4f& currPose, GlobalModel& globalModel, int maxDepthProcessed, float confidenceThreshold, int tick, int timeDelta, FillIn& fillIn, std::map<std::string, GPUTexture*>& textures)
{
    if(/*lastFrameRecovery*/false)
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 0,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    else
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 tick,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], false);
    fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], false);
    fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], false || /*frameToFrameRGB*/false);
}

Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

        
pangolin::OpenGlMatrix GetMvp(pangolin::OpenGlRenderState s_cam)
{

    pangolin::OpenGlMatrix view = s_cam.GetModelViewMatrix();
    pangolin::OpenGlMatrix projection = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix mvp =  projection * view;
    return mvp;
}

void visualize_mb(std::shared_ptr<Shader> program, pangolin::OpenGlMatrix mvp, const std::pair<GLuint, GLuint>& vbos, int vs)
{

    program->Bind();
    program->setUniform(Uniform("MVP", mvp));

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

    program->Unbind();

}