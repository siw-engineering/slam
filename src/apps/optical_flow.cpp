#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/optflow.hpp>
#include "../inputs/ros/DepthSubscriber.h"
#include "../inputs/ros/RGBSubscriber.h"

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
    RGBSubscriber* rgbsub;
    cv::Mat cur_dimg, cur_img, prev_dimg, prev_img;

	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);
    depthsub  = new DepthSubscriber("/X1/front/depth", nh);

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

        Mat flow(prev_img.size(), CV_32FC2);
        optflow::calcOpticalFlowSparseToDense(prev_img, cur_img, flow, 8, 128, 0.05f, true, 500.0f, 1.5f);
		Mat flow_parts[2];
		split(flow, flow_parts);
		Mat magnitude, angle, magn_norm;
		cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
		normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
		angle *= ((1.f / 360.f) * (180.f / 255.f));
		//build hsv image
		Mat _hsv[3], hsv, hsv8, bgr;
		_hsv[0] = angle;
		_hsv[1] = Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magn_norm;
		merge(_hsv, 3, hsv);
		hsv.convertTo(hsv8, CV_8U, 255.0);
		cvtColor(hsv8, bgr, COLOR_HSV2BGR);
		imshow("frame", cur_img);
		imshow("flow", bgr);
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
		break;

        ros::spinOnce();
		prev_img = cur_img;
		prev_dimg = cur_dimg;

    }

	return 0;
}