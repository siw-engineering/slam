#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 acolor;
layout (location = 2) in vec2 atexcord;

// out vec4 vPosition;
out vec3 color;
out vec2 texcord;

void main()
{
	// vPosition = position + .1;
	gl_Position = vec4(position,1);
	color = acolor;
	texcord = atexcord;
}