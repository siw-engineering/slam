#include <pangolin/pangolin.h>
#include <string>
#include <shaders/Shaders.h>


class Render
{
private:
	const float width;
	const float height;
	pangolin::OpenGlRenderState s_cam;
	pangolin::View d_cam;
	unsigned int m_RenderingID;

public:
	Render(const float w, const float h):
	width(w),
	height(h)
	{

		pangolin::CreateWindowAndBind("Main",width, height);
		glEnable(GL_DEPTH_TEST);

	    s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(width, height, 420, 420, width / 2.0f, height / 2.0f, 0.1, 1000),
	                                        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));

		pangolin::View& d_cam = pangolin::CreateDisplay()
		     .SetBounds(0.0, 1.0, 0.0, 1.0, -width/height)
		     .SetHandler(new pangolin::Handler3D(s_cam));

	}
	void VertexArray()
	{
		glGenVertexArrays(1, &m_RenderingID);

	}
	void addBuffer(int count, float size)
	{
	    glEnableVertexAttribArray(0);
	    glVertexAttribPointer(0, count, GL_FLOAT, GL_FALSE, size, (void*)0);

	}

	void vaoBind()
	{
		glBindVertexArray(m_RenderingID);
	}
	void vaoUnbind()
	{
		glBindVertexArray(0);
	}

	void VertexBuffer(const void *data, unsigned int size){

		glGenBuffers(1, &m_RenderingID);
	    glBindBuffer(GL_ARRAY_BUFFER, m_RenderingID);
	    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);	
	}

	void vboBind(){
		glBindBuffer(GL_ARRAY_BUFFER, m_RenderingID);
	}
	void vboUnbind(){
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void Texture()
	{
		glGenTextures(1, &m_RenderingID);
	}

	void TexGen(unsigned int width, unsigned int height, unsigned char* data)
	{

		glBindTexture(GL_TEXTURE_2D, m_RenderingID); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
		// set the texture wrapping parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   // set texture wrapping to GL_REPEAT (default wrapping method)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		// set texture filtering parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

		glBindTexture(GL_TEXTURE_2D, 0);

	}

	void textureBind()
	{
		glBindTexture(GL_TEXTURE_2D, m_RenderingID);
	}

	pangolin::OpenGlMatrix GetMvp()
	{

	    pangolin::OpenGlMatrix view = s_cam.GetModelViewMatrix();
	    pangolin::OpenGlMatrix projection = s_cam.GetProjectionMatrix();
	    pangolin::OpenGlMatrix model; //Entity->T_w_l;
	    pangolin::OpenGlMatrix mvp =  projection * view;

	    return mvp;
	}

	void bufferHandle(float vertices[], float size)
	{

		VertexArray();
	    VertexBuffer(vertices, size);
	    vaoBind();
	    addBuffer(3, sizeof(float) * 3);
	}

	void textureHandle()
	{
		VertexArray();

	    float vertices[] = {
	        // positions          // colors           // texture coords
	         0.5f,  0.5f, 0.0f, 1.0f, 1.0f, // top right
	         0.5f, -0.5f, 0.0f, 1.0f, 0.0f, // bottom right
	        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, // bottom left
	        -0.5f,  0.5f, 0.0f, 0.0f, 1.0f  // top left 
	    };

	    VertexBuffer(vertices, sizeof(vertices));
	    vaoBind();
		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);	    


	}

	void bufferUnbind()
	{
		vaoUnbind();
		vboUnbind();
	}

	void drawPrimitive(GLenum primitive, int count)
	{	
		glDrawArrays(primitive, 0, count);
	}
	void draw(std::string vert, std::string frag, GLenum primitive, int count)
	{

	    std::shared_ptr<Shader> program;
	    program =  std::shared_ptr<Shader>(loadProgramFromFile(vert, frag));


		// while( !pangolin::ShouldQuit() )
		// {
			pangolin::OpenGlMatrix mvp = GetMvp();
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		    program->Bind();
		    program->setUniform(Uniform("MVP", mvp));
		    drawPrimitive(primitive, count);
		 	program->Unbind();
		 	pangolin::FinishFrame();

		// }
	}

	void drawTexture(std::string vert, std::string frag, int height, int width,
		unsigned char* data)
	{

	    std::shared_ptr<Shader> program;
	    program =  std::shared_ptr<Shader>(loadProgramFromFile(vert, frag));

	    Texture();
	    TexGen(width, height, data);

	    while( !pangolin::ShouldQuit() )
	    {

	    	pangolin::OpenGlMatrix mvp = GetMvp();
	        program->Bind();
	        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	        textureBind();
	        glActiveTexture(GL_TEXTURE0);
	        program->setUniform(Uniform("MVP", mvp));

	     	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	    	program->Unbind();
	        pangolin::FinishFrame();


	}

}
	
	void glCoord(float data[], float vertices[], int size){

		for (int i = 0; i < size; i++){
			vertices[3 * i + 0] = data[i];
			vertices[3 * i + 1] = data[i + size];
			vertices[3 * i + 2] = data[i + 2 * (size)];
		}

	}

};
