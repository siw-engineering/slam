#include <cuda_runtime_api.h>
#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"
#include "Camera.h"
#include <iostream>
#include "cuda/cudafuncs.cuh"
#include "cuda/containers/device_array.hpp"
#include "RGBDOdometry.h"
#include "FillIn.h"
#include <unistd.h>
#include "Render.h"

using namespace std;


int main(int argc, char*argv[])
{

	int width, height, rows, cols;
	width = cols = 320;
	height = rows = 240;
	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity(), lastpose = Eigen::Matrix4f::Identity(), drawpose =Eigen::Matrix4f::Identity();
	Eigen::Matrix4f tinv;
	int count = 0;
	float depthCutOff, maxDepth;
	maxDepth = depthCutOff = 5;
	cv::Mat dimg, img;


	const int TEXTURE_DIMENSION = 3072;
	const int MAX_VERTICES = TEXTURE_DIMENSION * TEXTURE_DIMENSION;
	int VSIZE = 4;
	const int bufferSize = MAX_VERTICES * VSIZE * 3;
	int weightMultiplier = 1;

	Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	RGBDOdometry* rgbd_odom;
	depthsub  = new DepthSubscriber("/X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);


	GSLAM::CameraPinhole cam_model(320,240,277,277,160,120);
	CameraModel intr;
	intr.cx = cam_model.cx;
	intr.cy = cam_model.cy;
	intr.fx = cam_model.fx;
	intr.fy = cam_model.fy;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = cam_model.fx;
    K(1, 1) = cam_model.fy;
    K(0, 2) = cam_model.cx;
    K(1, 2) = cam_model.cy;
    Eigen::Matrix3f Kinv = K.inverse();

	rgbd_odom = new RGBDOdometry(width, height, (float)cam_model.cx, (float)cam_model.cy, (float)cam_model.fx, (float)cam_model.fy);

	//device arrays
	DeviceArray<float> rgb, rgb_prev, color_splat, vmaps_tmp, nmaps_tmp;
	DeviceArray2D<float> depth, depthf;
	DeviceArray2D<float> vmap, nmap, vmap_splat_prev, nmap_splat_prev/*, color_splat*/, vmap_test;
	DeviceArray2D<unsigned char> lastNextImage;
	DeviceArray2D<float> vmap_pi, nmap_pi, ct_pi;
	DeviceArray2D<unsigned int> index_pi;
    DeviceArray2D<unsigned int> time_splat;
	DeviceArray<float> model_buffer, unstable_buffer, model_buffer_rs;
	std::vector<DeviceArray2D<float>> depthPyr;

	depthPyr.resize(3);
	for (int i = 0; i < 3; ++i) 
	{
		int pyr_rows = height >> i;
		int pyr_cols = width >> i;
		depthPyr[i].create(pyr_rows, pyr_cols);
	}


	rgb.create(height*3*width);
	depth.create(height, width);
	lastNextImage.create(height, width);

	model_buffer.create(bufferSize);
	// initialize model buffers
	float* vertices = new float[bufferSize];
	memset(&vertices[0], 0, bufferSize);
	model_buffer.upload(&vertices[0], bufferSize);
	model_buffer_rs.upload(&vertices[0], bufferSize);
	delete[] vertices;

	int frame = 0;
	Render viewer(640, 480);


	//debug variables
	unsigned char* imageArray = new unsigned char[3*width*height];
  	pangolin::GlTexture imageTexture(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::View& d_image = pangolin::Display("image")
	.SetBounds(2/3.0f,1.0f,0,1/3.0f,640.0/480)
	.SetLock(pangolin::LockLeft, pangolin::LockTop);

	while (ros::ok())
	{
		img  = rgbsub->read();
		dimg = depthsub->read();
		if (dimg.empty() || img.empty())
		{
			ros::spinOnce();
			continue;	
		}

		// img.convertTo(img, CV_32FC3);
		// dimg.convertTo(dimg, CV_32FC1);
		// rgb.upload((float*)img.data, height*3*width);
		// depth.upload((float*)dimg.data, width*sizeof(float), height, width);
  		// computeBilateralFilter(depth, depthf, depthCutOff);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 		imageTexture.Upload(img.data,GL_RGB,GL_UNSIGNED_BYTE);
 		d_image.Activate();
 		imageTexture.RenderToViewport();

		lastpose = pose;
		frame++;		
		// count = 0;	// TO DO set lastpose
		ros::spinOnce();
		pangolin::FinishFrame();
	}

}