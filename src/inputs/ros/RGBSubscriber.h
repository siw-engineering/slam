#include <string>
#include <sensor_msgs/image_encodings.h>
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;

class RGBSubscriber
{
public:
	RGBSubscriber(std::string topic, ros::NodeHandle nh);
	~RGBSubscriber(){}
	void callback(const sensor_msgs::ImageConstPtr& msg);
	cv::Mat read();
private:
	image_transport::Subscriber sub;
	cv::Mat img;
};