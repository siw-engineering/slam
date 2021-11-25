#include "../cuda/convenience.cuh"

void computeCameraVelOF(float* angle_mat, float* mag_mat, float* dimg, float* ang_vel, float* cam_vel, float fx, int cols, int rows);