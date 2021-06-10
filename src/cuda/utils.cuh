#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include "containers/device_array.hpp"
#include "types.cuh"

void computeBilateralFilter(const DeviceArray2D<float>& depth,
                DeviceArray2D<float> & filtered,
                const float depthCutoff);

void fillinRgb(int width, int height, float* existingRgb, float* rawRgb, bool passthrough, float* dst);

void fillinVertex(const CameraModel& intr, int width, int height, DeviceArray2D<float>& existingVertex,
                DeviceArray2D<float>& rawDepth, bool passthrough, DeviceArray2D<float>& dst);

void fillinNormal(const CameraModel& intr, int width, int height, DeviceArray2D<float>& existingNormal,
                    DeviceArray2D<float>& rawDepth, bool passthrough, DeviceArray2D<float>& dst);

void ResizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output, const int factor);

void ResizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output, const int factor);

void Resize(const int height, const int width, float* src, unsigned char* dst, const int factor);

void SampleGraph(DeviceArray<float>& model_buffer, int count, DeviceArray<float>& sample_points, int* h_count);

#endif /* CUDA_CUDAFUNCS_CUH_ */

