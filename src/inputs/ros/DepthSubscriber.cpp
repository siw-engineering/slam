#include "DepthSubscriber.h"


DepthSubscriber::DepthSubscriber(std::string topic, ros::NodeHandle nh)
{
	image_transport::ImageTransport it(nh);
	sub = it.subscribe(topic, 1, &DepthSubscriber::callback, this);
}

void DepthSubscriber::callback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
		img = cv_ptr->image;
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
}

cv::Mat DepthSubscriber::read()
{
	return img;
}