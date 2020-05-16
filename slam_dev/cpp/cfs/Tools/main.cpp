#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "Core/Utils/Parse.h"
#include "Tools/LogReader.h"
#include "Tools/KlgLogReader.h"
#include "Core/GPUTexture.h"
#include "Core/Cuda/cudafuncs.cuh"
#include "Core/Utils/RGBDOdometry.h"
#include "Core/FrameData.h"
#include "Core/CoFusion.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


namespace py = pybind11;


class CFS
{
public:
	std::unique_ptr<LogReader> logReader;
	bool iclnuim;
	std::string logFile;
	std::string exportDir;
	bool exportSegmentation;
	float confGlobalInit, confObjectInit, icpErrThresh, covThresh, photoThresh, fernThresh;
	int timeDelta, icpCountThresh, start, end, preallocatedModelsCount;
	bool fillIn, openLoop, reloc, frameskip, quit, fastOdom, so3, frameToFrameRGB;
	unsigned modelSpawnOffset;
	float depthCutoff;
	float icpWeight;
	int destWidth, destHeight, factor;
	int width, height;
	unsigned char maskID = 0;	
	CoFusion* cf;
	FrameData frame;
	Eigen::Matrix4f* currentPose;
	CFS()
	{
		currentPose = 0;
	    factor = 8;
	    width = 640;
	    height = 480;

		// Parse::get().arg(argc, argv, "-l", logFile);
		logFile = "/home/christie/projects/work/siw/slam/co-fusion/dataset/teddy-handover.klg";

		Resolution::setResolution(640, 480);
		Intrinsics::setIntrinics(528, 528, 320, 240);

	    pangolin::Params windowParams;

	    windowParams.Set("SAMPLE_BUFFERS", 0);
	    windowParams.Set("SAMPLES", 0);


	    pangolin::CreateWindowAndBind("Main", 640, 480, pangolin::Params({{"scheme", "headless"}}));

		confObjectInit = 0.01f;
		confGlobalInit = 10.0f;
		icpErrThresh = 5e-05;
		covThresh = 1e-05;
		photoThresh = 115;
		fernThresh = 0.3095f;
		preallocatedModelsCount = 0;
		iclnuim = false;
		reloc = false;
		depthCutoff = 5;
		icpWeight = 10;
		frameToFrameRGB = false;
		fastOdom = false;
		modelSpawnOffset = 22;
		exportSegmentation = false;
		destWidth = 640/8;
		destHeight = 480/8;
		exportDir = "/tmp/";
		timeDelta = 200;  // Ignored, since openLoop
		icpCountThresh = 40000;
		start = 1;
		so3 = false;
		end = std::numeric_limits<unsigned short>::max();  // Funny bound, since we predict times in this format really!

	  	cf = 0;

		cf = new CoFusion(std::numeric_limits<int>::max() / 2, icpCountThresh, icpErrThresh, covThresh,
	                              false, iclnuim, reloc, photoThresh, confGlobalInit, confObjectInit, depthCutoff,
	                              icpWeight, fastOdom, fernThresh, so3, frameToFrameRGB, modelSpawnOffset,
	                              Model::MatchingType::Drost, exportDir, exportSegmentation);
		cf->preallocateModels(0);

		logReader = std::make_unique<KlgLogReader>(logFile, false);

	}
	bool hasmore()
	{
		return logReader->hasMore();
	}


	Eigen::Matrix4f next()
	{
		frame = logReader->getFrameData();
		logReader->getNext();
		cf->processFrame(logReader->getFrameData(), currentPose);
		return cf->getCurrPose();

	}
};

PYBIND11_MODULE(libsender, m){
	py::class_<CFS>(m, "CFS")
	.def(py::init<>())
	.def("hasmore", &CFS::hasmore)
	.def("next", &CFS::next);

}