#include "../cuda/convenience.cuh"
#include "../cuda/types.cuh"
#include "../cuda/containers/device_array.hpp"


void computeCameraVelOF(DeviceArray2D<float>& angle_mat, DeviceArray2D<float>& mag_mat, DeviceArray2D<float>& dimg, const float3 ang_vel, DeviceArray2D<float>& cam_vel, float fx, int cols, int rows);