#include "RGBSubscriber.h"


RGBSubscriber::RGBSubscriber(std::string topic, ros::NodeHandle nh)
{
	image_transport::ImageTransport it(nh);
	sub = it.subscribe(topic, 1, &RGBSubscriber::callback, this);
}

void RGBSubscriber::callback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		cv::Mat img = cv_ptr->image;
		ROS_INFO("img!");
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
}