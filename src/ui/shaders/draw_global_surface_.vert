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
out vec3 vColor0;


void main()
{
        gl_Position = MVP * vec4(position.xyz, 1.0);
        vColor0 = decodeColor(vColor.x);
}		
