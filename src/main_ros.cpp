#include "inputs/ros/RGBSubscriber.h"
// using namespace std;

int main(int argc, char **argv)
{
	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	RGBSubscriber rgbsub = new RGBSubscriber("")
	return 0;
}