#include "PoseSubscriber.h"


PoseSubscriber::PoseSubscriber(std::string topic, ros::NodeHandle nh)
{
	// sub = it.subscribe(topic, 1, &PoseSubscriber::callback, this);
	sub = nh.subscribe(topic, 1, &PoseSubscriber::callback, this);
}

void PoseSubscriber::callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    *x = msg->pose.pose.position.x;
    *y = msg->pose.pose.position.y;
    *z = msg->pose.pose.position.z;
    // *x = msg->pose.pose.position.x - 5.9999;
    // *y = msg->pose.pose.position.y + 4.9999;
    // *z = msg->pose.pose.position.z + 0.1322;
    *qw = msg->pose.pose.orientation.w;
    *qx = msg->pose.pose.orientation.x;
    *qy = msg->pose.pose.orientation.y;
    *qz = msg->pose.pose.orientation.z;
}

Eigen::MatrixXf PoseSubscriber::read()
{
    Eigen::Matrix3f mat3 = Eigen::Quaternionf(*qw, *qx, -*qz, *qy).toRotationMatrix();
    Eigen::Matrix4f mat4 = Eigen::Matrix4f::Identity();
    mat4.block(0,0,3,3) = mat3;
    mat4(0, 3) = *y;
    mat4(1, 3) = *z;
    mat4(2, 3) = *x;

    return mat4;
}