#include "ROSInputSubscriberSF.h"

ROSInputSubscriberSF::ROSInputSubscriberSF(std::string imu_topic, ros::NodeHandle nh)
{
	imu_sub = nh.subscribe(imu_topic, 1, &ROSInputSubscriberSF::imu_callback, this);
  robot_frame_id_ = "base_link";
  reference_frame_id_ = "map";

  x_ekf.setZero();
  x_ekf.qw() = 1.0;

  predictor.init(x_ekf);
  ekf.init(x_ekf);
  current_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("fused_pose", 10);
  

}

void ROSInputSubscriberSF::imu_callback(const sensor_msgs::Imu msg)
{

      current_stamp_ = msg.header.stamp;

      u.a_x() = msg.linear_acceleration.x;
      u.a_y() = msg.linear_acceleration.y;
      u.a_z() = 0.0;
      u.w_x() = 0;
      u.w_y() = msg.angular_velocity.y;
      u.w_z() = 0;
      // u.w_x() = 0.0;
      // u.w_y() = -msg.angular_velocity.y;
      // u.w_z() = -msg.angular_velocity.z;

      // x_ekf = sys.f(x_ekf, u);
      // x_pred = predictor.predict(sys, u);
      x_ekf = ekf.predict(sys, u);    

}


void ROSInputSubscriberSF::odom_observe(const geometry_msgs::PoseWithCovarianceStamped msg)
{

    PoseMeasure measure;
    measure.x() = msg.pose.pose.position.x;
    measure.y() = msg.pose.pose.position.y;
    measure.z() = 0.0;
    measure.qw() = msg.pose.pose.orientation.w;
    measure.qx() = msg.pose.pose.orientation.x;
    measure.qy() = msg.pose.pose.orientation.y;
    measure.qz() = msg.pose.pose.orientation.z;
    x_ekf = ekf.update(pose_measurement, measure);

	
}
	

void ROSInputSubscriberSF::broadcastPose()
{
    // x_ekf.setZero();
    // std::cout<<x_ekf.x()<<" "<<x_ekf.y()<<" "<<x_ekf.z()<<" "<<x_ekf.qx()<<" "<<x_ekf.qy()<<" "<<x_ekf.qz()<<" "<<x_ekf.qw()<<std::endl;
    geometry_msgs::PoseStamped current_pose_;
    current_pose_.header.stamp = current_stamp_;
    current_pose_.header.frame_id = reference_frame_id_;
    current_pose_.pose.position.x = x_ekf.x();
    current_pose_.pose.position.y = x_ekf.y();
    current_pose_.pose.position.z = x_ekf.z();
    current_pose_.pose.orientation.x = x_ekf.qx();
    current_pose_.pose.orientation.y = x_ekf.qy();
    current_pose_.pose.orientation.z = x_ekf.qz();
    current_pose_.pose.orientation.w = x_ekf.qw();
    current_pose_pub_.publish(current_pose_);
}


State ROSInputSubscriberSF::getPose()
{
    return x_ekf;
}