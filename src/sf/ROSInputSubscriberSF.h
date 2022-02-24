#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include "sensor_msgs/Imu.h"
#include <chrono>
#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman/UnscentedKalmanFilter.hpp>
#include "ekf/SystemModel.h"
#include "ekf/PoseMeasurementModel.h"


typedef float T;

// Some type shortcuts
typedef SF::State<T> State;
typedef SF::Control<T> Control;
typedef SF::SystemModel<T> SystemModel;
typedef SF::PoseMeasurement<T> PoseMeasure;
typedef SF::PoseMeasurementModel<T> PoseMeasureModel;


class ROSInputSubscriberSF
{
public:	
	ros::Subscriber init_pose_sub, imu_sub, gnss_sub ;
	ROSInputSubscriberSF(std::string imu_topic, ros::NodeHandle nh);
	void imu_callback(const sensor_msgs::Imu msg);
	void odom_observe(const geometry_msgs::PoseWithCovarianceStamped msg);
	State getPose();
	void broadcastPose();
private:
	std::string robot_frame_id_, reference_frame_id_;
	geometry_msgs::PoseStamped current_pose_;
    ros::Time current_stamp_;
    geometry_msgs::PoseStamped current_pose_odom_;
	ros::Publisher current_pose_pub_;
	Eigen::Vector3d odom_trans;
	State x_ekf;
	Control u;
	SystemModel sys;
	PoseMeasureModel pose_measurement;
	Kalman::ExtendedKalmanFilter<State> predictor;
	Kalman::ExtendedKalmanFilter<State> ekf;
  	State x_pred;


};