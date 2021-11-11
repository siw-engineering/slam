#include "IMUSubscriber.h"


IMUSubscriber::IMUSubscriber(std::string topic, ros::NodeHandle nh)
{
	sub = nh.subscribe(topic, 1, &IMUSubscriber::callback, this);

}

void IMUSubscriber::callback(const sensor_msgs::Imu::ConstPtr& msg)
{

    // geometry_msgs::Vector3Stamped acc_in, acc_out, w_in, w_out;

    // acc_in.vector.x = msg->linear_acceleration.x;
    // acc_in.vector.y = msg->linear_acceleration.y;
    // acc_in.vector.z = msg->linear_acceleration.z;
    // w_in.vector.x = msg->angular_velocity.x;
    // w_in.vector.y = msg->angular_velocity.y;
    // w_in.vector.z = msg->angular_velocity.z;
    // ros::Time time_point = ros::Time(
    // std::chrono::seconds(msg->header.stamp.sec) +
    // std::chrono::nanoseconds(msg->header.stamp.nsec));
    transformed_msg.header.stamp = msg->header.stamp;
    transformed_msg.angular_velocity.x = msg->angular_velocity.x;
    transformed_msg.angular_velocity.y = msg->angular_velocity.y;
    transformed_msg.angular_velocity.z = msg->angular_velocity.z;
    transformed_msg.linear_acceleration.x = msg->linear_acceleration.x;
    transformed_msg.linear_acceleration.y = msg->linear_acceleration.y;
    transformed_msg.linear_acceleration.z = msg->linear_acceleration.z;  

}

sensor_msgs::Imu IMUSubscriber::read()
{
    return transformed_msg;
}