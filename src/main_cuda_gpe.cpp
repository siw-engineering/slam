#include <cuda_runtime_api.h>
#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"
#include "Camera.h"
#include <iostream>
#include "cuda/cudafuncs.cuh"
#include "cuda/containers/device_array.hpp"
#include "RGBDOdometry.h"

using namespace std;


int main(int argc, char  *argv[])
{
	int width = 320;
	int height = 240;
	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f tinv;

	Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);

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

	DeviceArray<float> rgb, vmaps_tmp, nmaps_tmp;
	DeviceArray2D<float> depth;
	DeviceArray2D<float> vmap, nmap, vmap_dst, nmap_dst;

	rgb.create(height*3*width);
	depth.create(height, width);

	depthsub  = new DepthSubscriber("/X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);

	int i = 0;
	while (ros::ok())
	{
		img  = rgbsub->read();
		dimg = depthsub->read();
		if (dimg.empty() || img.empty())
		{
			ros::spinOnce();
			continue;	
		}
		img.convertTo(img, CV_32FC3);
		dimg.convertTo(img, CV_32FC1);
		rgb.upload((float*)img.data, height*3*width);
		depth.upload((float*)dimg.data, width*sizeof(float), height, width);
		if (!i)
		{
			rgbd_odom->initFirstRGB(rgb);
			createVMap(intr, depth, vmap, 100);
			createNMap(vmap, nmap);
		}
		tinv  = pose.inverse();
		splatDepthPredict(intr, height, width, tinv.data(), vmap, vmap_dst, nmap, nmap_dst);
		rgbd_odom->initICPModel(vmap_dst, nmap_dst, 100, pose);
		copyMaps(vmap, nmap, vmaps_tmp, nmaps_tmp);
		rgbd_odom->initRGBModel(rgb, vmaps_tmp);
		rgbd_odom->initICP(vmaps_tmp, nmaps_tmp, 100);
		rgbd_odom->initRGB(rgb, vmaps_tmp);
		transObject = pose.topRightCorner(3, 1);
		rotObject = pose.topLeftCorner(3, 3);
		rgbd_odom->getIncrementalTransformation(transObject, rotObject, false, 10, true, false, true, 0, 0);
		pose.topRightCorner(3, 1) = transObject;
		pose.topLeftCorner(3, 3) = rotObject;
		// std::cout<<"i :"<< i<< "\ntrans :"<<transObject<<std::endl<<"rot :"<<rotObject<<std::endl;
		std::cout<<"i :"<< i<< "\npose :"<<pose<<std::endl;

		i ++;
	}

	return 0;
}