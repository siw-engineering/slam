#include "Kalman.h"
#include "HungarianAlg.h"
#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>
#include <vector>
#include <time.h>
using namespace cv;
using namespace std;

class kalman_track
{
public:
	vector<Point2d> trace;
	time_t begin_time;
	int suspicious;
	int track_id;
	int misses; 
	Point2f prediction;
	TKalmanFilter* KF;
	kalman_track(int td, Point2f p, float dt, float acceleration);
	~kalman_track();
};




