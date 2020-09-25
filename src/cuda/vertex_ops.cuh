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

using namespace GSLAM;

void unproject(cv::Mat img, GSLAM::CameraPinhole* cam);


#endif /* CUDA_CUDAFUNCS_CUH_ */
