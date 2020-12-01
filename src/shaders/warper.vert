#version 330 core

layout (location = 0) in vec3 position;

// out vec4 vPosition;
out vec3 color;
out vec2 texcoord;


uniform sampler2D gSampler;
uniform sampler2D cSampler;
uniform vec4 cam; //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform int time;
uniform float maxDepth;

#include "surfels.glsl"
#include "geometry.glsl"
#include "color_encoding.glsl"


void main()
{
	// vPosition = position + .1;
	gl_Position = vec4(position,1);

}