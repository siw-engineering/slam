#version 330 core

in vec2 texcoord;
in vec3 color;

out vec4 FragColor;
uniform sampler2D s2d;

void main()
{
	FragColor = texture(s2d, texcoord);	
	// FragColor = vec4(0,1,0,1);
}
