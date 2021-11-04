#ifndef PoseMeasurementModel_HPP_
#define PoseMeasurementModel_HPP_
#include <kalman/LinearizedMeasurementModel.hpp>

namespace slam_robot
{
	template<typename T>
	class PoseMeasurement : public Kalman::Vector<T, 7>
	{
		public:
		KALMAN_VECTOR(PoseMeasurement, T , 7)
		
		static constexpr size_t X = 0;
		static constexpr size_t Y = 0;
		static constexpr size_t Z = 0;
		static constexpr size_t Q_X = 3;
		static constexpr size_t Q_Y = 4;
		static constexpr size_t Q_W = 6;
		static constexpr size_t Q_Z = 5;

		T x() const     { return (*this)[X]; }
		T y() const     { return (*this)[Y]; }
		T z() const     { return (*this)[Z]; }
		T qx() const    { return (*this)[Q_X];}
		T qy() const    { return (*this)[Q_Y];}
		T qz() const    { return (*this)[Q_Z];}
		T qw() const    { return (*this)[Q_W];}

		T& x()           { return (*this) [X];}
		T& y()           { return (*this) [Y];}
		T& z()           { return (*this) [Z];}
		T& qx()          { return (*this) [Q_X];}
		T& qy()          { return (*this) [Q_Y];}
		T& qz()          { return (*this) [Q_Z];}
		T& qw()          { return (*this) [Q_W];}

	};

	template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
	class PoseMeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, PoseMeasurement<T>, CovarianceBase>
	{
		public:
			typedef slam_robot::State<T> S;
			typedef slam_robot::PoseMeasurement<T> M;
			PoseMeasurementModel ()
			{
				//Setup noise jacobian
				this->H.setIdentity();
				this->V.setIdentity();
			}

			M h(const S& x) const
			{
				M measurement;
				measurement.x() = x.x();
				measurement.y() = x.y();
				measurement.z() = x.z();
				measurement.qx() = x.qx();
				measurement.qy() = x.qy();
				measurement.qz() = x.qz();
				measurement.qw() = x.qw();
				return measurement;
			}
		protected:
			void UpdateJacobians(const S& s)
			{
				this->H.setIdentity();
			}
	};
}
#endif
