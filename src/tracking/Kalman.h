#include <opencv2/opencv.hpp>
//#include "opencv/cv.h"
#include <opencv2/video/tracking.hpp>
#include <opencv4/opencv2/tracking/kalman_filters.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class TKalmanFilter
{
public:
	cv::KalmanFilter* kalman;
	double deltatime; //Time Increment
	Point3f LastResult;


	TKalmanFilter(Point3f p,float dt=0.2,float Accel_noise_mag=0.5);
	~TKalmanFilter();
	Point3f GetPrediction();
	Point3f Update(Point3f p, bool DataCorrect);
};

