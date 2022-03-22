#version 330 core

in vec3 vColor0;
out vec4 FragColor;


void main()
{
    FragColor = vec4(vColor0,1);
}
