#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"

#include "Camera.h"
#include "cuda/vertex_ops.cuh"
#include "cuda/containers/device_array.hpp"
#include "cuda/cudafuncs.cuh"
#include "RGBDOdometry.h"
#include "GPUTexture.h"

using namespace GSLAM;


int main(int argc, char **argv)
{

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	// GPUTexture gtex(320, 240, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
	RGBDOdometry odom(320,240,277,277,160,120);
	depthsub  = new DepthSubscriber("/X1/front/optical/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);


	cv::Mat img;
	GSLAM::CameraPinhole cam(320,240,277,277,160,120);
	std::vector<DeviceArray2D<unsigned char>> a; 
	// std::cout<<" cx :" <<cam.cx<<" cy :" <<cam.cy<<" fx :" <<cam.fx<<" fy :" <<cam.fy<<" fx_inv :" <<cam.fx_inv<<" fy_inv :" <<cam.fy_inv;

	cudaArray* ca;
	while (ros::ok())
	{
		// img = depthsub->read();
		img = rgbsub->read();

		if (img.empty()) 
		{
			ros::spinOnce();
			continue;
		}
		
		// gtex.texture->Upload(img.data,GL_RGB, GL_UNSIGNED_BYTE);

		uchar* camData = new uchar[img.total()*4];
		Mat continuousRGBA(img.size(), CV_8UC4, camData);
		cv::cvtColor(img, continuousRGBA, CV_BGR2RGBA, 4);
		// bool check;
		// check = cv::imwrite("src/MyImage.jpg", continuousRGBA);
		//     // if the image is not saved 
	 //    if (check == false) { 
	 //        std::cout << "Mission - Saving the image, FAILED" << std::endl; 
	  
	 //        // wait for any key to be pressed 
	 //        return -1; 
	 //    } 
	  
	 //    std::cout << "Successfully saved the image. " << std::endl; 
	 //    return 0;
		ca = rgb_texture_test(continuousRGBA);
		// odom.initFirstRGB(ca);


		delete(camData);
		ros::spinOnce();

	}
}