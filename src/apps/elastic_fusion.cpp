#include "inputs/KlgLogReader.h"
#include <libconfig.hh>
#include "../ui/EFGUI.h"

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

	EFGUI gui(width, height, intr);



	return 0;
}