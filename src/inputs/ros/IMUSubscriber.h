#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Imu.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tf2/transform_datatypes.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <chrono>
#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman/UnscentedKalmanFilter.hpp>
#include "../../sf/ekf/SystemModel.h"
#include "../../sf/ekf/PoseMeasurementModel.h"



class IMUSubscriber
{
public:
	IMUSubscriber(std::string topic, ros::NodeHandle nh);
	~IMUSubscriber(){}
	void callback(const sensor_msgs::Imu::ConstPtr& msg);
	sensor_msgs::Imu read();
private:
	ros::Subscriber sub;
	sensor_msgs::Imu transformed_msg;
	tf2_ros::Buffer tfbuffer_;

};