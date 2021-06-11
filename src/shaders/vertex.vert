#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 MVP;
uniform int r;
uniform int g;
uniform int b;

flat out int rout;
flat out int gout;
flat out int bout;

void main()
{
	// vMVP = MVP;
	rout = r;
	bout = b;
	gout = g;
	gl_Position = MVP * vec4(aPos, 1.0);
	// gl_PointSize  = 20;
}
