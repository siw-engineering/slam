#ifndef MODEL_H_
#define MODEL_H_

#include "../gl/FeedbackBuffer.h"
#include "IndexMap.h"
#include "../gl/FillIn.h"
#include <Eigen/LU>
#include <list>
#include "../odom/RGBDOdometryef.h"

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
		inline const Eigen::Matrix4f& getPose() const { return pose; }

		inline IndexMap& getIndexMap() { return indexMap; }
		void fuse(  const int & time,
					GPUTexture * rgb,
					GPUTexture * depthRaw,
					GPUTexture * depthFiltered,
					const float depthCutoff,
					const float confThreshold,
					const float weighting);

		void clean( const int & time,
					const float confThreshold,
					std::vector<float> & graph,
					const int timeDelta,
					const float maxDepth,
					const bool isFern);

		// odometry
		void initFirstRGB(GPUTexture* rawRGB){
			frameToModel.initFirstRGB(rawRGB);
		}

		void performTracking(GPUTexture* rawRGB, GPUTexture* filteredDepth, bool shouldFillIn, const float maxDepthProcessed,
							bool rgbOnly, float icpWeight, bool pyramid, bool fastOdom);

		// model projection
		inline void predictIndices( const int & time,
									const float depthCutoff,
									const int timeDelta){
			indexMap.predictIndices(getPose(), time, getModel(), depthCutoff, timeDelta);
		}

		inline void combinedPredict(const float depthCutoff,
									const float confThreshold,
									const int time,
									const int maxTime,
									const int timeDelta,
									IndexMap::Prediction predictionType){
			indexMap.combinedPredict(getPose(), getModel(), depthCutoff, confThreshold, time, maxTime, timeDelta, predictionType);
		}

		void performFillIn(GPUTexture* rawRGB, GPUTexture* rawDepth/*, bool frameToFrameRGB, bool lost*/);

		unsigned int lastCount();

		Eigen::Vector4f * downloadMap();

		inline GPUTexture* getRGBProjection() { return indexMap.imageTex(); }
		inline GPUTexture* getVertexConfProjection() { return indexMap.vertexTex(); }
		inline GPUTexture* getNormalProjection() { return indexMap.normalTex(); }
		inline GPUTexture* getTimeProjection() { return indexMap.timeTex(); }

		inline GPUTexture* getFillInImageTexture() { return &(fillIn.imageTexture); }
		inline GPUTexture* getFillInVertexTexture() { return &(fillIn.vertexTexture); }
		inline GPUTexture* getFillInNormalTexture() { return &(fillIn.normalTexture); }

		int width, height, numPixels;
		CameraModel intr;
	private:

		Eigen::Matrix4f pose;

		RGBDOdometryef frameToModel;

		//First is the vbo, second is the fid
		std::pair<GLuint, GLuint> * vbos;
		int target, renderSource;

		IndexMap indexMap;
		FillIn fillIn;

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