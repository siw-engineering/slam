#include "vertex_ops.cuh"

__global__
void unproject_kernel(GSLAM::CameraPinhole* cam, unsigned char *depth, unsigned char* d_3d_points)
{

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int idy = threadIdx.y + (blockIdx.y * blockDim.y);
	int id;
	id = idx * idy;
	// int z = depth[id];
	printf("cam cx : ");

	// printf("cam cx : %d\n",cam->cx);
	// d_3d_points[id] = (idx-cam->cx)*cam->fx_inv;
	// d_3d_points[id+(rows*cols)] = (idy-cam->cy)*cam->fy_inv;
	// d_3d_points[id+2*(rows*cols)] = 1;

}


void unproject(cv::Mat img, GSLAM::CameraPinhole* cam)
{

	// uchar4 **ddepth, **input_image;
	// unsigned char *hdepth;
	unsigned char *d_depth_image, *d_3d_points;
	int rows, cols;
	rows = img.rows;
	cols = img.cols;
	size_t totalpixels = rows*cols;
	const dim3 dimGrid((int)ceil((cols)/16), (int)ceil((rows)/16));
	const dim3 dimBlock(16, 16);
	// *input_image = (uchar4 *)img.ptr<uchar4 *>(0);
	// cudaMalloc(ddepth, sizeof(uchar4) * totalpixels * CHANNELS);
	// cudaMemcpy(*ddepth, *input_image, sizeof(uchar4) * totalpixels * CHANNELS, cudaMemcpyHostToDevice);

	unsigned char* depth_image = (unsigned char*)img.data;
	cudaMalloc((void **)&d_depth_image, sizeof(unsigned char) * totalpixels );
	cudaMalloc((void **)&d_3d_points, sizeof(unsigned char) * totalpixels * 3 );
	cudaMemcpy(d_depth_image, depth_image, sizeof(unsigned char) * totalpixels , cudaMemcpyHostToDevice);
	unproject_kernel<<<dimGrid,dimBlock>>>(cam, d_depth_image, d_3d_points);

	// cudaFree(ddepth);
}