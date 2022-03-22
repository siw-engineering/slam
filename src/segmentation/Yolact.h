#ifndef YOLACTTEST_H_
#define YOLACTTEST_H_
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <iostream>
#include <pangolin/gl/gl.h>


using namespace std;
using namespace cv;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};

class Yolact{
public:
	Yolact();


    // struct ProcessingChainData
    // {
    //     cv::Mat img;
    //     std::vector<cv::Rect> faces, faces2;
    //     cv::Mat gray, smallImg;
    // };
    ncnn::Net yolact;
    std::vector<Object> objects;
	cv::Mat processFrame(int fd);
    // void run(cv::Mat &img2, tbb::concurrent_bounded_queue<ProcessingChainData *> &frameQueue1);
    int detect_yolact(std::vector<Object>& objects, int imgShareableHandle);
    inline float intersection_area(const Object& a, const Object& b);
    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold);
    cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::vector<Object>& track_objects);
    void computeBBox(std::vector<Object> objects, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr,  int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height);

};
#endif /* YOLACTTEST_H_ */