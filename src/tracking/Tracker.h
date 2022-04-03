#include "multitracker.h"
#include "../segmentation/Yolact.h"

class Tracker
{

	std::vector<Object> track_objects;
	vector<kalman_track*> tracks;
	float acceleration =0.2;
	int max_distance= 60;
	int max_misses= 60;
	int max_trace= 1;
	float dt = 0.2;
	int NextID = 0 ;
	time_t start_time;
	std::map<int,Eigen::Vector3f> idx_to_rgb;

public:
	int* obj_tid; // use only after Update() call
	Tracker();
	void Update(vector<Point2f>& detections);
	void Update();
	void Update(std::vector<Object> objects, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr,  int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height);
	float distance(int x1, int y1, int x2, int y2);
	float encodeColor(Eigen::Vector3f c);
	Eigen::Vector3f decodeColor(float c);





};
