#version 330 core

flat in int rout;
flat in int gout;
flat in int bout;
out vec4 FragColor;  
// uniform sampler2D Tex;
void main()
{
    // FragColor = texture(Tex, TexCoord);
    FragColor = vec4(rout, gout, bout, 1);
}