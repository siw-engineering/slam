#include "ros/ros.h"
#include "std_msgs/String.h"
#include <nav_msgs/Odometry.h>
#include <Eigen/Core>
#include <Eigen/Geometry>


class PoseSubscriber
{
public:
	PoseSubscriber(std::string topic, ros::NodeHandle nh);
	~PoseSubscriber(){}
	void callback(const nav_msgs::Odometry::ConstPtr &msg);
	Eigen::MatrixXf read();
private:
	ros::Subscriber sub;
	double *x = new double;
	double *y = new double;
	double *z = new double;
	double *qw = new double;
	double *qx = new double;
	double *qy = new double;
	double *qz = new double;
};