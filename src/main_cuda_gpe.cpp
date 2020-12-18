#include <cuda_runtime_api.h>
#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"
#include "Camera.h"
#include <iostream>
#include "cuda/cudafuncs.cuh"
#include "cuda/containers/device_array.hpp"
#include "RGBDOdometry.h"

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
	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	cv::Mat dimg, img;
	RGBDOdometry* rgbd_odom;

	GSLAM::CameraPinhole cam_model(320,240,277,277,160,120);
	CameraModel intr;
	intr.cx = cam_model.cx;
	intr.cy = cam_model.cy;
	intr.fx = cam_model.fx;
	intr.fy = cam_model.fy;

	rgbd_odom = new RGBDOdometry(width, height, (float)cam_model.cx, (float)cam_model.cy, (float)cam_model.fx, (float)cam_model.fy);

	DeviceArray<float> rgb;
	DeviceArray2D<unsigned char> intesity_map;
	DeviceArray2D<float> depth;
	
	DeviceArray2D<float> vmap, nmap, vmap_dst, nmap_dst;

	rgb.create(height*3*width);
	intesity_map.create(height, width);
	depth.create(height, width);

	depthsub  = new DepthSubscriber("/X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);

	float vmap_host[width*height*3];
	while (ros::ok())
	{
		img  = rgbsub->read();
		img.convertTo(img, CV_32FC3);
		dimg = depthsub->read();
		if (dimg.empty() || img.empty())
		{
			ros::spinOnce();
			continue;	
		}

		rgb.upload((float*)img.data, height*3*width);
		// imageBGRToIntensityDM(rgb, intesity_map);
		depth.upload(dimg.data, width*sizeof(float), height, width);
		createVMap(intr, depth, vmap, 100);
		createNMap(vmap, nmap);
		splatDepthPredict(intr, height, width, pose.data(), vmap, vmap_dst, nmap, nmap_dst);

		// vmap.download(&vmap_host[0], width*sizeof(float));

	}

	return 0;
}