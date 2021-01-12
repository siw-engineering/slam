#ifndef CUDA_CUDAFUNCS_CUH_
#define CUDA_CUDAFUNCS_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include <stdio.h>
#include<stdlib.h> 
#include "Camera.h"
#include <opencv2/core/core.hpp>
#include "Point.h"
#include "cuda/containers/device_array.hpp"
#include "cuda/convenience.cuh"
#include "cuda/cudafuncs.cuh"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"


using namespace GSLAM;

void unproject(cv::Mat img, GSLAM::CameraPinhole cam);
// void rgb_texture_test(unsigned char* input, unsigned char* output, int width, int height, int widthstep);
void rgb_texture_test(cv::Mat img);


#endif /* CUDA_CUDAFUNCS_CUH_ */
