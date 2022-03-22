#version 330 core

layout (location = 0) in vec4 position;

out vec4 vColor0;

uniform mat4 MVP;
uniform mat4 pose;


vec3 decodeColor(float c)
{
    vec3 col;
    col.x = float(int(c) >> 16 & 0xFF) / 255.0f;
    col.y = float(int(c) >> 8 & 0xFF) / 255.0f;
    col.z = float(int(c) & 0xFF) / 255.0f;
    return col;
}

void main()
{
    gl_Position = MVP * pose * (vec4(position.xyz, 1.0));
    vColor0 = vec4(decodeColor(position.w),1);
}		
