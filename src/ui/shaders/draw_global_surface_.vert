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

#version 330 core

#include "color.glsl"

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 vColor;


uniform mat4 MVP;
uniform bool maskDraw;
uniform mat4 t_inv;
uniform usampler2D maskSampler;
out vec3 vColor0;


void main()
{
        gl_Position = MVP * vec4(position.xyz, 1.0);
        if (maskDraw)
        {
                vec3 localPos = (t_inv * vec4(position.xyz, 1.0f)).xyz;
                // xn = f*x/z + cx
                float x = ((528 * localPos.x) / localPos.z) + 320;
                float y = ((528 * localPos.y) / localPos.z) + 240;

                float halfPixelx = 0.5 * (1.0f / 640);
                float halfPixely = 0.5 * (1.0f / 480);

                float xn = (x / 640) + halfPixelx;
                float yn = (y / 480) + halfPixely;

                int maskValue = int(textureLod(maskSampler, vec2(xn, yn), 0));

                if (maskValue>1)
                        vColor0 = (decodeColor(vColor.x)*0.6) + (vec3(10, 0, 0)*(1-0.6));
                else
                        vColor0 = decodeColor(vColor.x);

        }
        else
        {
                vColor0 = decodeColor(vColor.x);
        }
        
}
