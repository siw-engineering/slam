#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"

#include "Camera.h"
#include "cuda/vertex_ops.cuh"
#include "cuda/containers/device_array.hpp"
#include "cuda/cudafuncs.cuh"
#include "RGBDOdometry.h"

using namespace GSLAM;


int main(int argc, char **argv)
{

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	RGBDOdometry odom(320,240,277,277,160,120);
	depthsub  = new DepthSubscriber("/X1/front/optical/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);

	cv::Mat img;
	GSLAM::CameraPinhole cam(320,240,277,277,160,120);
	std::vector<DeviceArray2D<unsigned char>> a; 
	// std::cout<<" cx :" <<cam.cx<<" cy :" <<cam.cy<<" fx :" <<cam.fx<<" fy :" <<cam.fy<<" fx_inv :" <<cam.fx_inv<<" fy_inv :" <<cam.fy_inv;


	while (ros::ok())
	{
		// img = depthsub->read();
		img = rgbsub->read();

		if (img.empty()) 
		{
			ros::spinOnce();
			continue;
		}
		
		uchar* camData = new uchar[img.total()*4];
		Mat continuousRGBA(img.size(), CV_32FC4, camData);
		cv::cvtColor(img, continuousRGBA, CV_BGR2RGBA, 4);

		rgb_texture_test(continuousRGBA);
		
		delete(camData);
		ros::spinOnce();

	}
}