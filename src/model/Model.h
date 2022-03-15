#ifndef MODEL_H_
#define MODEL_H_

#include "../gl/FeedbackBuffer.h"
#include "IndexMap.h"
#include <Eigen/LU>
#include <list>

class Model;
typedef std::shared_ptr<Model> ModelPointer;
typedef std::list<ModelPointer> ModelList;
typedef ModelList::iterator ModelListIterator;

class Model
{
	public:
		Model(int width, int height, CameraModel intr, std::string shader_dir);
		virtual ~Model();

		void initialise(const FeedbackBuffer & rawFeedback,
		                const FeedbackBuffer & filteredFeedback);

		static const int TEXTURE_DIMENSION;
		static const int MAX_VERTICES;
		static const int NODE_TEXTURE_DIMENSION;
		static const int MAX_NODES;

		void renderPointCloud(pangolin::OpenGlMatrix mvp,
		                      const float threshold,
		                      const bool drawUnstable,
		                      const bool drawNormals,
		                      const bool drawColors,
		                      const bool drawPoints,
		                      const bool drawWindow,
		                      const bool drawTimes,
		                      const int time,
		                      const int timeDelta);

		const std::pair<GLuint, GLuint> & getModel();

		void fuse(const Eigen::Matrix4f & pose,
		          const int & time,
		          GPUTexture * rgb,
		          GPUTexture * depthRaw,
		          GPUTexture * depthFiltered,
		          GPUTexture * indexMap,
		          GPUTexture * vertConfMap,
		          GPUTexture * colorTimeMap,
		          GPUTexture * normRadMap,
		          const float depthCutoff,
		          const float confThreshold,
		          const float weighting);

		void clean(const Eigen::Matrix4f & pose,
		           const int & time,
		           GPUTexture * indexMap,
		           GPUTexture * vertConfMap,
		           GPUTexture * colorTimeMap,
		           GPUTexture * normRadMap,
		           GPUTexture * depthMap,
		           const float confThreshold,
		           std::vector<float> & graph,
		           const int timeDelta,
		           const float maxDepth,
		           const bool isFern);

		unsigned int lastCount();

		Eigen::Vector4f * downloadMap();

		int width, height, numPixels;
		CameraModel intr;
	private:
		//First is the vbo, second is the fid
		std::pair<GLuint, GLuint> * vbos;
		int target, renderSource;

		const int bufferSize;

		GLuint countQuery;
		unsigned int count;

		std::shared_ptr<Shader> initProgram;
		std::shared_ptr<Shader> drawProgram;
		std::shared_ptr<Shader> drawSurfelProgram;

		//For supersample fusing
		std::shared_ptr<Shader> dataProgram;
		std::shared_ptr<Shader> updateProgram;
		std::shared_ptr<Shader> unstableProgram;
		pangolin::GlRenderBuffer renderBuffer;

		//We render updated vertices vec3 + confidences to one texture
		GPUTexture updateMapVertsConfs;

		//We render updated colors vec3 + timestamps to another
		GPUTexture updateMapColorsTime;

		//We render updated normals vec3 + radii to another
		GPUTexture updateMapNormsRadii;

		//16 floats stored column-major yo'
		GPUTexture deformationNodes;

		GLuint newUnstableVbo, newUnstableFid;

		pangolin::GlFramebuffer frameBuffer;
		GLuint uvo;
		int uvSize;
};
#endif /* MODEL_H_ */