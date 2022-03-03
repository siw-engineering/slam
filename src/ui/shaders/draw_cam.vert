#version 330 core

layout (location = 0) in vec4 position;
out vec3 vColor0;

uniform mat4 MVP;
uniform mat4 pose;


void main()
{
        gl_Position = MVP * pose * vec4(position.xyz, 1.0);
        vColor0 = vec3(0,1,0);
}		
