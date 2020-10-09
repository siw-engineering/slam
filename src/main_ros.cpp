#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"

#include "Camera.h"
#include "cuda/vertex_ops.cuh"
#include "cuda/containers/device_array.hpp"
#include "cuda/cudafuncs.cuh"
#include "RGBDOdometry.h"
// #include "GPUTexture.h"

using namespace GSLAM;


int main(int argc, char **argv)
{

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	// GPUTexture gtex(320, 240, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
	RGBDOdometry odom(320,240,277,277,160,120);
	depthsub  = new DepthSubscriber("/ROBOTIKA_X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/ROBOTIKA_X1/front/image_raw", nh);


	cv::Mat img;
	GSLAM::CameraPinhole cam(320,240,277,277,160,120);
	std::vector<DeviceArray2D<unsigned char>> a; 
	// std::cout<<" cx :" <<cam.cx<<" cy :" <<cam.cy<<" fx :" <<cam.fx<<" fy :" <<cam.fy<<" fx_inv :" <<cam.fx_inv<<" fy_inv :" <<cam.fy_inv;

	cv::Mat s_img;
	while (ros::ok())
	{
		// img = depthsub->read();
		img = rgbsub->read();

		if (img.empty()) 
		{
			ros::spinOnce();
			continue;
		}
		
		unsigned char *camData = new unsigned char[img.total()*4];
		unsigned char *h_img = new unsigned char[img.total()*4];
		cv::Mat continuousRGBA(img.size(), CV_8UC4, camData);
		cv::Mat s_img(img.size(), CV_8UC4, h_img);
		// cv::cvtColor(img, continuousRGBA, CV_BGR2RGBA, 4);
		img.convertTo(continuousRGBA, CV_8UC4);
		
		unsigned char *input, *ouput;
		input = (unsigned char *)continuousRGBA.data;
		ouput = (unsigned char *)s_img.data;

		int width = img.cols;
		int height = img.rows;
		int nchannels = 4;

		int widthstep = (width*sizeof(unsigned char)*nchannels);
		rgb_texture_test(input, ouput, width, height, widthstep);
		// bool check;

		// float *fdata = new float[img.total()*4];
		// cv::Mat f_img_mat(img.size(), CV_32FC4, fdata);

		// s_img.convertTo(f_img_mat,CV_32FC4);

		// check = cv::imwrite("src/MyImage.jpg", im);
		//    if (check == false) { 
		//        std::cout << "Mission - Saving the image, FAILED" << std::endl; 

		//        // wait for any key to be pressed 
		//        return -1; 
		//    } 
	 // 	std::cout << "Successfully saved the image. " << std::endl; 
		// // odom.initFirstRGB(ca);


		delete(camData);
		delete(h_img);

		ros::spinOnce();

	}
}