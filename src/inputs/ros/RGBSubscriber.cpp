#include "RGBSubscriber.h"


RGBSubscriber::RGBSubscriber(std::string topic, int argc, char** argv):
{

	ros::init(argc, argv, "image_sub");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	rgb_sub = it.subscribe(rgb_topic, 1, &RGBSubscriber::callback, this);
	ROS_INFO("Init");

}

void callback()
{
	
}