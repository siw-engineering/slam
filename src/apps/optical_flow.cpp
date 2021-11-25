#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/optflow.hpp>
#include "../inputs/ros/DepthSubscriber.h"
#include "../inputs/ros/RGBSubscriber.h"
#include "../inputs/ros/IMUSubscriber.h"
#include "../of/utils.cuh"
#include "../cuda/containers/device_array.hpp"

using namespace std;


int main(int argc, char *argv[])
{
	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
    RGBSubscriber* rgbsub;
    IMUSubscriber* imusub;

    cv::Mat cur_dimg, cur_img, prev_dimg, prev_img;
    DeviceArray2D<float> curr_depth_d2d, angle_d2d, mag_d2d, cam_vel_d2d;

	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);
    depthsub  = new DepthSubscriber("/X1/front/depth", nh);
    imusub  = new IMUSubscriber("/X1/imu/data", nh);

    int height = 240;
    int width = 320;

    while (ros::ok())
    {
		cur_img  = rgbsub->read();
		cur_dimg = depthsub->read();

       if (cur_img.empty() || cur_dimg.empty())
        {
            ros::spinOnce();
            continue;   
        }
        if (prev_img.empty() || prev_dimg.empty())
        {
			prev_img = cur_img;
			prev_dimg = cur_dimg;
        	ros::spinOnce();
            continue;  
        }

        curr_depth_d2d.create(height, width);
        angle_d2d.create(height, width);
        mag_d2d.create(height, width);
        cam_vel_d2d.create(height, width);


        cv::Mat flow(prev_img.size(), CV_32FC2);
        cv::Mat cam_vel(prev_img.size(), CV_32FC1);

        optflow::calcOpticalFlowSparseToDense(prev_img, cur_img, flow, 8, 128, 0.05f, true, 500.0f, 1.5f);
		cv::Mat flow_parts[2];
		split(flow, flow_parts);
		cv::Mat magnitude, angle, magn_norm;
		cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
		normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

		sensor_msgs::Imu imu_msg;
		imu_msg = imusub->read();
		Eigen::Vector3f angular_vel(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);
		
		curr_depth_d2d.upload((float*)cur_dimg.data, width*sizeof(float), height, width);
		angle_d2d.upload((float*)angle.data, width*sizeof(float), height, width);
		mag_d2d.upload((float*)magn_norm.data, width*sizeof(float), height, width);


		float3 device_angl_vel;
		device_angl_vel = *reinterpret_cast<float3*>(angular_vel.data());
		
		computeCameraVelOF(angle_d2d, mag_d2d, curr_depth_d2d, device_angl_vel, cam_vel_d2d, 277.2, width, height);
		
		float* cam_vel_hst = new float[height*width];
		cam_vel_d2d.download(cam_vel_hst, width*sizeof(float));
		cv::Mat cam_vel_mat(height, width, CV_32FC1, cam_vel_hst);

		// std::cout<<cam_vel<<std::endl;
		// double minVal;     double maxVal;          cv::minMaxLoc(magn_norm, &minVal, &maxVal); std::cout<<"min:"<<minVal<<" max:"<<maxVal<<std::endl;
	

		// std::vector<cv::Mat> channels;
		// split(cam_vel, channels);
		// cv::Scalar m = mean(channels[0]);
		// std::cout<<"cam_vel x : "<<m[0]<<std::endl;	

		angle *= ((1.f / 360.f) * (180.f / 255.f));
		//build hsv image
		cv::Mat _hsv[3], hsv, hsv8, bgr;
		_hsv[0] = angle;
		_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magn_norm;

		
		merge(_hsv, 3, hsv);
		hsv.convertTo(hsv8, CV_8U, 255.0);
		cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
		imshow("frame", cur_img);
		imshow("flow", bgr);
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
		break;


		// std::vector<cv::Mat> channels;
		// split(cam_vel, channels);
		// cv::Scalar m = mean(channels[0]);
		// std::cout<<"cam_vel x : "<<m[0]<<std::endl;

		// std::cout<<angle<<std::endl;
        ros::spinOnce();
		prev_img = cur_img;
		prev_dimg = cur_dimg;

    }

	return 0;
}