#include "inputs/KlgLogReader.h"
#include <libconfig.hh>
#include "../ui/EFGUI.h"
#include "../odom/RGBDOdometryef.h"
#include "../gl/FeedbackBuffer.h"

using namespace libconfig;

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
  	float confidence, depth, icp, icpErrThresh, covThresh, photoThresh, fernThresh, timeDelta, icpCountThresh, depthCutoff, maxDepthProcessed;
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

	//Camera Params
	CameraModel intr(640, 480, 528, 528, 320, 240);
	root["camera"].lookupValue("width", intr.width);
	root["camera"].lookupValue("height", intr.height);
	root["camera"].lookupValue("fx", intr.fx);
	root["camera"].lookupValue("fy", intr.fy);
	root["camera"].lookupValue("cx", intr.cx);
	root["camera"].lookupValue("cy", intr.cy);

	//GUI
	int width, height;
	root["gui"].lookupValue("width", width);
	root["gui"].lookupValue("height", height);

	EFGUI gui(width, height, intr.cx, intr.cy, intr.fx, intr.fy);
	RGBDOdometryef frameToModel(width, height, intr.cx,intr.cy, intr.fx, intr.fy);

	std::map<std::string, GPUTexture*> textures;

	//createtextures
	textures[GPUTexture::RGB] = new GPUTexture(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
    textures[GPUTexture::DEPTH_RAW] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
    textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT, false, true);
    textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    textures[GPUTexture::DEPTH_NORM] = new GPUTexture(width, height, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT, true);


	return 0;
}