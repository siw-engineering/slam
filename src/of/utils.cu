#include "utils.cuh"

__global__ void computeCameraVelOFKernel(float* angle_mat, float* mag_mat, float* dimg, int cols, int rows)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < cols && y < rows)
    {
    	
    }

}

void computeCameraVelOF(float* angle_mat, float* mag_mat, float* dimg, int cols, int rows)
{

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(cols, block.x);
	grid.y = getGridDim(rows, block.y);

    computeCameraVelOFKernel<<<grid, block>>>(angle_mat, mag_mat, dimg, rows, cols);
    cudaSafeCall(cudaGetLastError());
}