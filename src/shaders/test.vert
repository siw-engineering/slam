#version 330 core

layout (location = 0) in vec2 texcoord;
out vec4 vPosition;
out vec4 vColor;
out vec4 vNormRad;
out float zVal;
out mat4 vMVP;

// out vec4 vPosition;


uniform sampler2D gSampler;
uniform sampler2D cSampler;
uniform vec4 cam; //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform mat4 MVP;


#include "surfels.glsl"
#include "geometry.glsl"
#include "color_encoding.glsl"


void main()
{
	// vPosition = position + .1;
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    vPosition = vec4(getVertex(texcoord.xy, x, y, cam, gSampler), 1);
    vPosition = vec4(texcoord.x,texcoord.y,vPosition.z,1);
    vColor = textureLod(cSampler, texcoord.xy, 0.0);
    
    vec3 vNormLocal = getNormal(vPosition.xyz, texcoord.xy, x, y, cam, gSampler);
    vNormRad = vec4(vNormLocal, getRadius(vPosition.z, vNormLocal.z));
    vMVP = MVP;
    gl_Position = MVP * vPosition;
}