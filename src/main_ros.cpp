#include "inputs/ros/DepthSubscriber.h"
#include "Camera.h"
#include "cuda/vertex_ops.cuh"
using namespace GSLAM;

int main(int argc, char **argv)
{

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	depthsub  = new DepthSubscriber("/X1/front/optical/depth", nh);
	cv::Mat img;
	GSLAM::CameraPinhole cam(320,240,277,277,160,120);
	// std::cout<<" cx :" <<cam.cx<<" cy :" <<cam.cy<<" fx :" <<cam.fx<<" fy :" <<cam.fy<<" fx_inv :" <<cam.fx_inv<<" fy_inv :" <<cam.fy_inv;

	while (ros::ok())
	{
		img = depthsub->read();
		if (img.empty()) 
		{
			ros::spinOnce();
			continue;
		}
		unproject(img, cam);
		ros::spinOnce();

	}
}