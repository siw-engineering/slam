#include "inputs/ros/RGBSubscriber.h"
#include "Camera.h"
#include "cuda/vertex_ops.cuh"
using namespace GSLAM;

int main(int argc, char **argv)
{
	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	RGBSubscriber* rgbsub;
	rgbsub  = new RGBSubscriber("/X1/front/image_raw", nh);
	cv::Mat img;

	GSLAM::CameraPinhole cam(320,240,277,277,160,120);
	GSLAM::Point2d p2(50,50);
	GSLAM::Point3d p3;
	p3 = cam.UnProject(p2);
	while (ros::ok())
	{
		img = rgbsub->read();
		std::cout<<img.size()<<std::endl;
		if (!img.isContinuous()) 
		{
			std::cerr << "Images aren't continuous!! Exiting." << std::endl;
			continue;
		}
		// unproject(img, cam);
		ros::spinOnce();
	}
}