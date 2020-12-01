#version 330 core

in vec2 texcoord;
in vec4 vColor0;

out vec4 FragColor;
uniform sampler2D s2d;

void main()
{
	FragColor = vColor0;	
	// FragColor = vec4(0,1,0,1);
}
