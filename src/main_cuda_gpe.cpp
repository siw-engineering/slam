#include <cuda_runtime_api.h>
#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"
#include "Camera.h"
#include <iostream>
#include "cuda/cudafuncs.cuh"
#include "cuda/containers/device_array.hpp"

using namespace std;

class BGR8
{
	uint8_t r,g,b;
	BGR8(){}
	~BGR8(){}
};

int main(int argc, char  *argv[])
{
	int width = 320;
	int height = 240;

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	cv::Mat dimg, img;

	GSLAM::CameraPinhole cam_model(320,240,277,277,160,120);
	CameraModel intr;
	intr.cx = cam_model.cx;
	intr.cy = cam_model.cy;
	intr.fx = cam_model.fx;
	intr.fy = cam_model.fy;

	DeviceArray2D<BGR8> rgb;
	DeviceArray2D<float> depth;
	
	DeviceArray2D<float> vmap;

	rgb.create(height, width);
	depth.create(height, width);

	depthsub  = new DepthSubscriber("/X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);

	float vmap_host[width*height*3];
	while (ros::ok())
	{
		img  = rgbsub->read();
		dimg = depthsub->read();
		if (dimg.empty() || img.empty())
		{
			ros::spinOnce();
			continue;	
		}
		// std::cout<<"\nw->"<<img.cols<<std::endl<<img.rows;
		// rgb.upload(img.data, width*sizeof(BGR8), height, width);
		depth.upload(dimg.data, width*sizeof(float), height, width);
		createVMap(intr, depth, vmap, 100);
		vmap.download(&vmap_host[0], width*sizeof(float));

	}

	return 0;
}