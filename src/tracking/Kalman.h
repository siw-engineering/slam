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
	Point2f LastResult;



	TKalmanFilter(Point2f p,float dt=0.2,float Accel_noise_mag=0.5);
	~TKalmanFilter();
	Point2f GetPrediction();
	Point2f Update(Point2f p, bool DataCorrect);
};

