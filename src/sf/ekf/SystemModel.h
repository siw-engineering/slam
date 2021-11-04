#ifndef SYSTEMMODEL_HPP_
#define SYSTEMMODEL_HPP_

#include <kalman/LinearizedSystemModel.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
namespace slam_robot
{
	template<typename T>
	class State : public Kalman::Vector<T, 10>
	{
		public:
		KALMAN_VECTOR(State, T, 10)

		static constexpr size_t X = 0;
		static constexpr size_t Y = 1;
		static constexpr size_t Z = 2;
		static constexpr size_t Q_X = 3;
		static constexpr size_t Q_Y = 4;
		static constexpr size_t Q_Z = 5;
		static constexpr size_t Q_W = 6;
		static constexpr size_t V_X = 7;
		static constexpr size_t V_Y = 8;
		static constexpr size_t V_Z = 9;

		T x() const     { return (*this)[X]; }
		T y() const     { return (*this)[Y]; }
		T z() const     { return (*this)[Z]; }
		T qx() const    { return (*this)[Q_X];}
		T qy() const    { return (*this)[Q_Y];}
		T qz() const    { return (*this)[Q_Z];}
		T qw() const    { return (*this)[Q_W];}
		T vx() const     { return (*this)[V_X]; }
		T vy() const     { return (*this)[V_Y]; }
		T vz() const     { return (*this)[V_Z]; }

		T& x()           { return (*this) [X];}
		T& y()           { return (*this) [Y];}
		T& z()           { return (*this) [Z];}
		T& qx()          { return (*this) [Q_X];}
		T& qy()          { return (*this) [Q_Y];}
		T& qz()          { return (*this) [Q_Z];}
		T& qw()          { return (*this) [Q_W];}
		T& vx()           { return (*this) [V_X];}
		T& vy()           { return (*this) [V_Y];}
		T& vz()           { return (*this) [V_Z];}
	};

	template<typename T>
	class Control : public Kalman::Vector<T, 6>
	{
		public:
		KALMAN_VECTOR(Control, T, 6)
		//! Acceleration
		static constexpr size_t A_X = 0;
		static constexpr size_t A_Y = 1;
		static constexpr size_t A_Z = 2;
		//! Angular Rate 
		static constexpr size_t W_X = 3;
		static constexpr size_t W_Y = 4;
		static constexpr size_t W_Z = 5;

		T a_x()    const { return (*this)[ A_X ];}
		T a_y()    const { return (*this)[ A_Y ];}
		T a_z()    const { return (*this)[ A_Z ];}
		T w_x()    const { return (*this)[ W_X ];}
		T w_y()    const { return (*this)[ W_Y ];}
		T w_z()    const { return (*this)[ W_Z ];}

		T& a_x()   {return (*this) [ A_X ];}
		T& a_y()   {return (*this) [ A_Y ];}
		T& a_z()   {return (*this) [ A_Z ];}
		T& w_x()   {return (*this) [ W_X ];}
		T& w_y()   {return (*this) [ W_Y ];}
		T& w_z()   {return (*this) [ W_Z ];}
	};

	template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
	class SystemModel : public Kalman::LinearizedSystemModel<State<T>, Control<T>, CovarianceBase>
	{
		public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		typedef slam_robot::State<T> S;
		typedef slam_robot::Control<T> C;
		// Definition of (non-linear) state transition function
		S f(const S& x, const C& u) const
		{
			//! Predicted state vector after transition
			S x_;
			Eigen::Quaterniond Qwb;
			Eigen::Vector3d Pwb;
			Eigen::Vector3d Vw;
			double dt = 0.01;
			Qwb.x() = x.qx();
			Qwb.y() = x.qy();
			Qwb.z() = x.qz();
			Qwb.w() = x.qw();
			Pwb<<x.x(),x.y(),x.z();
			Vw<<x.vx(), x.vy(), x.vz();
			Eigen::Vector3d imu_gyro(u.w_x(), u.w_y(), u.w_z());
			Eigen::Vector3d imu_acc(u.a_x(), u.a_y(), u.a_z());

			Eigen::Quaterniond dq;
			Eigen::Vector3d half_newOriention = imu_gyro * dt / 2.0;
			dq.w() = 1;
			dq.x() = half_newOriention.x();
			dq.y() = half_newOriention.y();
			dq.z() = half_newOriention.z();

			Eigen::Vector3d acc_w = Qwb * imu_acc;
			Qwb = Qwb.normalized() * dq.normalized();
			Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;
			Vw = Vw + acc_w * dt;

			x_.x() = Pwb(0);
			x_.y() = Pwb(1);
			x_.z() = Pwb(2);
			x_.qw() = Qwb.w();
			x_.qx() = Qwb.x();
			x_.qy() = Qwb.y();
			x_.qz() = Qwb.z();
			x_.vx() = Vw.x();
			x_.vy() = Vw.y();
			x_.vz() = Vw.z();

		return x_;
		}

		protected:  
		void updataJacbians(const S& x, const C& u)
		{
			this->F.setIdentity();
			this->W.setIdentity();
		}
	};
}
#endif

