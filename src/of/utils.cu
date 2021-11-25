#include "utils.cuh"

__global__ void computeCameraVelOFKernel(PtrStepSz<float> angle_mat, PtrStepSz<float> mag_mat, PtrStepSz<float> dimg, const float3 ang_vel, PtrStepSz<float> cam_vel, float f, int cols, int rows)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int depthCutoff = 3;

    if (x < cols && y < rows)
    {
        float Z = dimg.ptr(y)[x];

        if(Z != 0 && Z < depthCutoff)
        {
            
            float theta = angle_mat.ptr(y)[x]*3.14159265/180;
            float mag = mag_mat.ptr(y)[x];
            float u_ = mag * cos(theta);
            float v_ = mag * sin(theta);

            /*
            image jacobian
            [-f/Z     0    u/Z   uv/f   -(f+u^2/f)   v
               0    -f/Z   v/Z   f+v^2/f   -uv/f    -u]
            */

            float Ju_w = (x*y/f)*ang_vel[0] - (f + x*x/f)*ang_vel[1] + y*ang_vel[2];
            float Jv_w = (f + y*y/f)*ang_vel[0] - (x*y/f)*ang_vel[1] - x*ang_vel[2];

            
            cam_vel.ptr(y)[x] = 2;
            // cam_vel[y*cols + x] = Z * ( Ju_w - u_) / f;
            // cam_vel[y*cols*2 + x*2 + 1] = Z * ( Jv_w - v_) / f;
        }
        else
            cam_vel.ptr(y)[x] = -1;

    }

}

void computeCameraVelOF(DeviceArray<float>2D& angle_mat, DeviceArray<float>2D& mag_mat, DeviceArray<float>2D& dimg, const float3 ang_vel, DeviceArray<float>2D& cam_vel, float fx, int cols, int rows)
{

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);


    computeCameraVelOFKernel<<<grid, block>>>(angle_mat, mag_mat, dimg, ang_vel, cam_vel, fx, cols, rows);
    cudaSafeCall(cudaGetLastError());
}