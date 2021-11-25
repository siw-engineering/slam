#include "../cuda/convenience.cuh"
#include "../cuda/types.cuh"


void computeCameraVelOF(DeviceArray<float>2D& angle_mat, DeviceArray<float>2D& mag_mat, DeviceArray<float>2D& dimg, const float3 ang_vel, DeviceArray<float>2D& cam_vel, float fx, int cols, int rows);