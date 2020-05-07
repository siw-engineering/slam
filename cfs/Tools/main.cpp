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

int main(int argc, char* argv[])
{

	unsigned char maskID = 0;
	CoFusion* cf;
  	std::unique_ptr<LogReader> logReader;
	std::string logFile;
 	std::string empty;
	FrameData frame;
  	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f* currentPose = 0;

	Parse::get().arg(argc, argv, "-l", logFile);

	Resolution::setResolution(640, 480);
	Intrinsics::setIntrinics(528, 528, 320, 240);
  	cf = new CoFusion();
	
	cf->preallocateModels(0);

	RGBDOdometry* frametToModel;
	frametToModel = new RGBDOdometry(Resolution::getInstance().width(), Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
               Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(), maskID);

	std::map<std::string, GPUTexture*> textures;
	textures[GPUTexture::RGB] = new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);

	logReader = std::make_unique<KlgLogReader>(logFile, Parse::get().arg(argc, argv, "-f", empty) > -1);

	while (logReader->hasMore())
	{

		frame = logReader->getFrameData();
		logReader->getNext();
		cf->processFrame(logReader->getFrameData(), currentPose);
		// std::cout<<cf->getCurrPose();

	}
	return 0;
}
