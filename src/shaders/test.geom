#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;


in vec4 vPosition[];
in vec4 vColor[];
in vec4 vNormRad[];
in mat4 vMVP[];

out vec3 vColor0;

void main()
{
	vColor0 = vNormRad[0].xyz;
	vec3 x = normalize(vec3((vNormRad[0].y - vNormRad[0].z), -vNormRad[0].x, vNormRad[0].x)) * vNormRad[0].w * 1.41421356;
        
	vec3 y = cross(vNormRad[0].xyz, x);
	gl_Position = vMVP[0] * vec4(vPosition[0].xyz + x, 1.0);
    EmitVertex();

	gl_Position = vMVP[0] * vec4(vPosition[0].xyz + y, 1.0);
	EmitVertex();

	gl_Position = vMVP[0] * vec4(vPosition[0].xyz - y, 1.0);
	EmitVertex();

	gl_Position = vMVP[0] * vec4(vPosition[0].xyz - x, 1.0);
	EmitVertex();

}