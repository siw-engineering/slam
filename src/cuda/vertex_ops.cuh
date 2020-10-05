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

using namespace GSLAM;

void unproject(cv::Mat img, GSLAM::CameraPinhole cam);
void rgb_texture_test(cv::Mat img);

#endif /* CUDA_CUDAFUNCS_CUH_ */
