/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef FEEDBACKBUFFER_H_
#define FEEDBACKBUFFER_H_

#include "shaders/Shaders.h"
#include "Uniform.h"
#include "Vertex.h"
#include <pangolin/gl/gl.h>
#include <pangolin/display/opengl_render_state.h>
#include "cuda/types.cuh"


#include "Defines.h"

class FeedbackBuffer
{
    public:
        FeedbackBuffer(std::shared_ptr<Shader> program, int width, int height, CameraModel intr);
        virtual ~FeedbackBuffer();

        std::shared_ptr<Shader> program;

        void compute(pangolin::GlTexture * color,
                     pangolin::GlTexture * depth,
                     const int & time,
                     const float depthCutoff);

        EFUSION_API void render(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f & pose, const bool drawNormals, const bool drawColors);

        EFUSION_API static const std::string RAW, FILTERED;

        GLuint vbo;
        GLuint fid;
        int width, height, numPixels;
        CameraModel intr;

    private:
        std::shared_ptr<Shader> drawProgram;
        GLuint uvo;
        GLuint countQuery;
        const int bufferSize;
        unsigned int count;
};

#endif /* FEEDBACKBUFFER_H_ */
