#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"

#include "Camera.h"
#include "cuda/vertex_ops.cuh"
#include "cuda/containers/device_array.hpp"
#include "cuda/cudafuncs.cuh"
#include "RGBDOdometry.h"
// #include "GPUTexture.h"

#include <librealsense2/rs.hpp> 

using namespace GSLAM;


int main(int argc, char **argv)
{
    string inputStream = argv[1];

    RGBDOdometry odom(320,240,277,277,160,120);

    if (inputStream == "real")
    {
     int inWidtih = 640, inHeight = 480, inFps = 30;
	 rs2::pipeline pipe;
     rs2::config cfg;
     //cfg.enable_device_from_file("/home/sathish/Documents/20201113_111309.bag");
     cfg.enable_stream(RS2_STREAM_COLOR, inWidth, inHeight, RS2_FORMAT_RGB8,inFps);
     cfg.enable_stream(RS2_STREAM_DEPTH, inWidth, inHeight, RS2_FORMAT_Z16, inFps);
     cfg.enable_stream(RS2_STREAM_INFRARED, 1, inWidth, inHeight, RS2_FORMAT_Y8, inFps);
     pipe.start(cfg);

     rs2::align align(RS2_STREAM_INFRARED);
    } 

    else{
    	ros::init(argc, argv, "test_node");
    	ros::NodeHandle nh;
    	DepthSubscriber* depthsub;
        RGBSubscriber* rgbsub;
	// GPUTexture gtex(320, 240, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
	
	    depthsub  = new DepthSubscriber("/ROBOTIKA_X1/front/depth", nh);
	    rgbsub = new RGBSubscriber("/ROBOTIKA_X1/front/image_raw", nh);


	    
        } 

    cv::Mat img;
	GSLAM::CameraPinhole cam(320,240,277,277,160,120);
	std::vector<DeviceArray2D<unsigned char>> a; 
	// std::cout<<" cx :" <<cam.cx<<" cy :" <<cam.cy<<" fx :" <<cam.fx<<" fy :" <<cam.fy<<" fx_inv :" <<cam.fx_inv<<" fy_inv :" <<cam.fy_inv;

	int h, w = 512;

	cv::Mat s_img;


	while (true)
	{
		// img = depthsub->read();

		Mat img(640, 480, CV_8UC3);

		if (inputStream == "real")
		{
			auto frames = pipe.wait_for_frames();
	        auto aligned_frames = align.process(frames);
            
            rs2::depth_frame current_depth_frame = aligned_frames.get_depth_frame();
            rs2::video_frame current_color_frame = aligned_frames.get_color_frame();

            Mat image(Size(640, 480), CV_8UC3, (void*)current_color_frame.get_data() , Mat::AUTO_STEP);
        


            Mat depth_image(Size(640, 480), CV_16UC1, (void*)current_depth_frame.get_data() , Mat::AUTO_STEP);

		    cv::cvtColor(image, image, CV_BGR2RGB);

		    //imshow("COLOR Image", image);


		    Mat hsv, color_image, dark_image;

		//cv::cvtColor(image, image, CV_BGR2BGRA);
		    cv::cvtColor(image,hsv,CV_BGR2HSV);

		    const auto result = cv::mean(hsv);

		    cout <<"Brightness : "<<result[2]<<endl;

		    //Mat img(640, 480, CV_8UC3);


		    if (result[2]<128)
		   {   
		    rs2::video_frame current_infrared_frame = aligned_frames.get_infrared_frame(1);
			Mat infrared_image(Size(640, 480), CV_8UC1, (void*)current_infrared_frame.get_data() , Mat::AUTO_STEP);
			//imshow("INFRARED Image", infrared_image);
			img = infrared_image.clone();

		   }
		   else{img = image.clone();}
		         //imshow("DEPTH Image", depth_image);

		}
		

		else{
		    img = rgbsub->read();

		    if (img.empty()) 
		    {
			    ros::spinOnce();
			    continue;
		    }
		
		// unsigned char *camData = new unsigned char[img.total()*4];
		// unsigned char *h_img = new unsigned char[img.total()*4];
		// cv::Mat continuousRGBA(img.size(), CV_8UC4, camData);
		// cv::Mat s_img(img.size(), CV_8UC4, h_img);
		// // cv::cvtColor(img, continuousRGBA, CV_BGR2RGBA, 4);
		// img.convertTo(continuousRGBA, CV_8UC4);
		
		// unsigned char *input, *ouput;
		// input = (unsigned char *)continuousRGBA.data;
		// ouput = (unsigned char *)s_img.data;

		    
	        }


        cv::cvtColor(img, img, CV_BGR2BGRA);


		rgb_texture_test(img);




		ros::spinOnce();

	}
}