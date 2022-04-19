#include "Kalman.h"
#include "HungarianAlg.h"
#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>
#include <vector>
#include <time.h>
#include <pangolin/pangolin.h>

using namespace cv;
using namespace std;

class kalman_track
{
public:
	vector<Point3d> trace;
	time_t begin_time;
	int suspicious;
	int track_id;
	int misses; 
	GLfloat bbox_vertices_ptr[8*4];
	GLushort bbox_elements_ptr[24];
	int ec;
	Point3f prediction;
	TKalmanFilter* KF;
	kalman_track(int td, Point3f p, float dt, float acceleration);
	~kalman_track();
};




