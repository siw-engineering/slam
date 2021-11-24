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


using namespace std;
using namespace cv;
class Yolact{
public:
	Yolact();

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
        std::vector<float> maskdata;
        cv::Mat mask;
    };
    struct ProcessingChainData
    {
        cv::Mat img;
        std::vector<cv::Rect> faces, faces2;
        cv::Mat gray, smallImg;
    };

    std::vector<Object> objects;
	void processFrame(cv::Mat img1);
    void run(cv::Mat &img2, tbb::concurrent_bounded_queue<ProcessingChainData *> &frameQueue1);
    int detect_yolact(const cv::Mat& bgr, std::vector<Object>& objects);
    inline float intersection_area(const Object& a, const Object& b);
    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold);
    Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);


};
#endif /* YOLACTTEST_H_ */