#include "../ui/GUI.h"
#include "inputs/RawLogReader.h"
#include "../gl/ComputePack.h"
#include "../gl/FeedbackBuffer.h"
#include "../gl/Vertex.h"




void model_initialise(std::shared_ptr<Shader> initProgram, const FeedbackBuffer & rawFeedback, const FeedbackBuffer & filteredFeedback, const std::pair<GLuint, GLuint>& vbos, GLuint & countQuery, unsigned int & count)
{

    initProgram->Bind();
    glBindBuffer(GL_ARRAY_BUFFER, rawFeedback.vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

    glBindBuffer(GL_ARRAY_BUFFER, filteredFeedback.vbo);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos.second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos.first);

    glBeginTransformFeedback(GL_POINTS);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

    //It's ok to use either fid because both raw and filtered have the same amount of vertices
    glDrawTransformFeedback(GL_POINTS, rawFeedback.fid);

    glEndTransformFeedback();

    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

    glDisable(GL_RASTERIZER_DISCARD);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    initProgram->Unbind();
    glFinish();
}



int main(int argc, char const *argv[])
{

	std::string gl_shaders = "/home/developer/slam/src/gl/shaders/";
	std::string model_shaders = "/home/developer/slam/src/model/shaders/";

	int width, height;
	width = 640;
	height = 480;
	float depthCutoff = 3;
	float maxDepthProcessed = 20;
	int tick = 1;

	CameraModel intr(528,528,320,240,640,480);
    GUI gui("/home/developer/slam/src/ui/shaders/");
   	
	std::shared_ptr<Shader> initProgram;
   	initProgram = loadProgramFromFile("init_unstable.vert", model_shaders);
	
	std::map<std::string, GPUTexture*> textures;
	std::map<std::string, ComputePack*> computePacks;
	std::map<std::string, FeedbackBuffer*> feedbackBuffers;

	textures[GPUTexture::RGB] = new GPUTexture(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
	textures[GPUTexture::DEPTH_RAW] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT, false, true);
	textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
	textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
	textures[GPUTexture::DEPTH_NORM] = new GPUTexture(width, height, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT, true);
	textures[GPUTexture::MASK] = new GPUTexture(width, height, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, false, true);

	// //createcompute
	computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_NORM]->texture, width, height);
	computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_FILTERED]->texture, width, height);
	computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_METRIC]->texture, width, height);
	computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, width, height);

	//createfeedbackbuffers
	feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom", gl_shaders), width, height, intr);
	feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom", gl_shaders), width, height, intr);


	LogReader * logReader;
	logReader = new RawLogReader("/home/developer/datasets/dyson_lab.klg", false, width, height);

	int bufferSize;
	std::pair<GLuint, GLuint> * vbos;

    vbos = new std::pair<GLuint, GLuint>;
	bufferSize = 3072 * 3072 * Vertex::SIZE;

    float *vertices = new float[bufferSize];
    memset(&vertices[0], 0, bufferSize);
	GLuint countQuery;
	unsigned int count = 0;

    glGenTransformFeedbacks(1, &vbos[0].second);
    glGenBuffers(1, &vbos[0].first);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    initProgram->Bind();

    int locInit[3] =
    {
        glGetVaryingLocationNV(initProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(initProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(initProgram->programId(), "vNormRad0"),
    };

    glTransformFeedbackVaryingsNV(initProgram->programId(), 3, locInit, GL_INTERLEAVED_ATTRIBS);

    glGenQueries(1, &countQuery);

    //Empty both transform feedbacks
    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].first);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, 0);

    glEndTransformFeedback();

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    glDisable(GL_RASTERIZER_DISCARD);

    initProgram->Unbind();

    delete [] vertices;

	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
	while (logReader->hasMore())
	{
		logReader->getNext();
		textures[GPUTexture::DEPTH_RAW]->texture->Upload(logReader->depth, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
		textures[GPUTexture::RGB]->texture->Upload(logReader->rgb, GL_RGB, GL_UNSIGNED_BYTE);

		std::vector<Uniform> uniformsfd;
		uniformsfd.push_back(Uniform("cols", (float)width));
		uniformsfd.push_back(Uniform("rows", (float)height));
		uniformsfd.push_back(Uniform("maxD", depthCutoff));
		computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniformsfd);

		std::vector<Uniform> uniforms;
		uniforms.push_back(Uniform("maxD", depthCutoff));
		computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
		computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);

		feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC]->texture, tick, maxDepthProcessed);
		feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, tick, maxDepthProcessed);

		model_initialise(initProgram, *feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED], vbos[0], countQuery, count);

		gui.renderModel(vbos[0],  Vertex::SIZE, pose);

	}

	return 0;
}