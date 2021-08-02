#include "inputs/RawLogReader.h"
#include <libconfig.hh>
#include "../ui/EFGUI.h"
#include "../odom/RGBDOdometryef.h"
#include "../gl/FeedbackBuffer.h"
#include "../gl/ComputePack.h"
#include "../gl/FillIn.h"
#include "../model/GlobalModel.h"
#include "../model/IndexMap.h"
#include "../gl/Vertex.h"

using namespace libconfig;


void predict(IndexMap& indexMap, Eigen::Matrix4f& currPose, GlobalModel& globalModel, int maxDepthProcessed, float confidenceThreshold, int tick, int timeDelta, FillIn& fillIn, std::map<std::string, GPUTexture*>& textures)
{
    if(/*lastFrameRecovery*/false)
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 0,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    else
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 tick,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], false);
    fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], false);
    fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], false || /*frameToFrameRGB*/false);
}

Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}


int main(int argc, char const *argv[])
{

	Config cfg;
	try
	{
		cfg.readFile("/home/developer/slam/src/configs/ef.cfg");
	}
	catch(const FileIOException &fioex)
	{
		std::cerr << "I/O error while reading file." << std::endl;
		return(EXIT_FAILURE);
	}

	// ElasticFusion Params
  	float confidence, depth, icp, icpErrThresh, covThresh, photoThresh, fernThresh, depthCutoff, maxDepthProcessed;
	int timeDelta, icpCountThresh;
	const Setting& root = cfg.getRoot();
	root["ef"].lookupValue("confidence", confidence);
	root["ef"].lookupValue("depth", depth);
	root["ef"].lookupValue("icp", icp);
	root["ef"].lookupValue("icpErrThresh", icpErrThresh);
	root["ef"].lookupValue("covThresh", covThresh);
	root["ef"].lookupValue("photoThresh", photoThresh);
	root["ef"].lookupValue("fernThresh", fernThresh);
	root["ef"].lookupValue("timeDelta", timeDelta);
	root["ef"].lookupValue("icpCountThresh", icpCountThresh);
	root["ef"].lookupValue("depthCutoff", depthCutoff);
	root["ef"].lookupValue("maxDepthProcessed", maxDepthProcessed);

	std::cout<<confidence<<std::endl<<depth<<std::endl<<icp<<std::endl<<icpErrThresh<<std::endl<<covThresh<<std::endl<<photoThresh<<std::endl<<fernThresh<<std::endl<<depthCutoff<<std::endl<<maxDepthProcessed;
	//Camera Params
	CameraModel intr(0,0,0,0,0,0);
	root["camera"].lookupValue("width", intr.width);
	root["camera"].lookupValue("height", intr.height);
	root["camera"].lookupValue("fx", intr.fx);
	root["camera"].lookupValue("fy", intr.fy);
	root["camera"].lookupValue("cx", intr.cx);
	root["camera"].lookupValue("cy", intr.cy);

	//GUI
	float width, height;
	root["gui"].lookupValue("width", width);
	root["gui"].lookupValue("height", height);



	EFGUI gui(width, height, intr.cx, intr.cy, intr.fx, intr.fy);
	RGBDOdometryef frameToModel(width, height, intr.cx,intr.cy, intr.fx, intr.fy);


	//data
  	std::string logFile;
	root["data"].lookupValue("path", logFile);
	LogReader * logReader;
    logReader = new RawLogReader(logFile, false, width, height);

	std::map<std::string, GPUTexture*> textures;
	std::map<std::string, ComputePack*> computePacks;
	std::map<std::string, FeedbackBuffer*> feedbackBuffers;


	//createtextures
	textures[GPUTexture::RGB] = new GPUTexture(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
    textures[GPUTexture::DEPTH_RAW] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
    textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT, false, true);
    textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    textures[GPUTexture::DEPTH_NORM] = new GPUTexture(width, height, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT, true);

    //createcompute
    computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom", "/home/developer/slam/src/gl/shaders/"), textures[GPUTexture::DEPTH_NORM]->texture, width, height);
    computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom", "/home/developer/slam/src/gl/shaders/"), textures[GPUTexture::DEPTH_FILTERED]->texture, width, height);
    computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom", "/home/developer/slam/src/gl/shaders/"), textures[GPUTexture::DEPTH_METRIC]->texture, width, height);
    computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom", "/home/developer/slam/src/gl/shaders/"), textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, width, height);

    //createfeedbackbuffers
    feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom", "/home/developer/slam/src/gl/shaders/"), width, height, intr);
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom", "/home/developer/slam/src/gl/shaders/"), width, height, intr);
	

    IndexMap indexMap(width, height, intr);
    GlobalModel globalModel(width, height, intr);
    FillIn fillIn(width, height, intr);


    int tick = 1;
    Eigen::Matrix4f currPose = Eigen::Matrix4f::Identity();

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

		if (tick == 1)
		{
			feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC]->texture, tick, maxDepthProcessed);
   			feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, tick, maxDepthProcessed);

	        globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);
	        frameToModel.initFirstRGB(textures[GPUTexture::RGB]);	
		}
		else
		{
       		Eigen::Matrix4f lastPose = currPose;
       		bool trackingOk = frameToModel.lastICPError < 1e-04;
            bool shouldFillIn = true;
            frameToModel.initICPModel(shouldFillIn ? &fillIn.vertexTexture : indexMap.vertexTex(),
                                      shouldFillIn ? &fillIn.normalTexture : indexMap.normalTex(),
                                      maxDepthProcessed, currPose);
            frameToModel.initRGBModel((shouldFillIn || false) ? &fillIn.imageTexture : indexMap.imageTex());
            frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
            frameToModel.initRGB(textures[GPUTexture::RGB]);
            Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);
            frameToModel.getIncrementalTransformation(trans, rot, false, 0.3, true, false, true);
            currPose.topRightCorner(3, 1) = trans;
            currPose.topLeftCorner(3, 3) = rot;

			Eigen::Matrix4f diff = currPose.inverse() * lastPose;
			Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
			Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

			//Weight by velocity
			float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());
			float largest = 0.01;
			float minWeight = 0.5;
       		std::vector<float> rawGraph;

			if(weighting > largest)
				weighting = largest;

			weighting = std::max(1.0f - (weighting / largest), minWeight) * 1;
	   		predict(indexMap, currPose, globalModel, maxDepthProcessed, confidence, tick, timeDelta, fillIn, textures);
			Eigen::Matrix4f recoveryPose = currPose;

			if (trackingOk)
	        {
	            indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);
	            globalModel.fuse(currPose,
	                             tick,
	                             textures[GPUTexture::RGB],
	                             textures[GPUTexture::DEPTH_METRIC],
	                             textures[GPUTexture::DEPTH_METRIC_FILTERED],
	                             indexMap.indexTex(),
	                             indexMap.vertConfTex(),
	                             indexMap.colorTimeTex(),
	                             indexMap.normalRadTex(),
	                             maxDepthProcessed,
	                             confidence,
	                             weighting);

	            indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);

	            globalModel.clean(currPose,
	                              tick,
	                              indexMap.indexTex(),
	                              indexMap.vertConfTex(),
	                              indexMap.colorTimeTex(),
	                              indexMap.normalRadTex(),
	                              indexMap.depthTex(),
	                              confidence,
	                              rawGraph,
	                              timeDelta,
	                              maxDepthProcessed,
	                              false);
	        }
        }

	    predict(indexMap, currPose, globalModel, maxDepthProcessed, confidence, tick, timeDelta, fillIn, textures);
	    gui.render(globalModel.model(), Vertex::SIZE);

		tick++;

	}

	return 0;
}