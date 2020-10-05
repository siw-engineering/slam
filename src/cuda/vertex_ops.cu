#include "vertex_ops.cuh"



__global__
void unproject_kernel(unsigned char *depth, double* d_3d_points, int rows, int cols, double cx, double cy, double fx, double fy, double fx_inv, double fy_inv)
{

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int idy = threadIdx.y + (blockIdx.y * blockDim.y);
	int id;
	if ((idx < rows ) && (idy < cols))
	{
		// add depth info
		id = idx * idy;
		double z = (double)depth[id];
		d_3d_points[id] = (idx-cx)*fx_inv*z;
		d_3d_points[id+(rows*cols)] = (idy-cy)*fy_inv*z;
		d_3d_points[id+2*(rows*cols)] = z;
		// printf("--%f--\n",(cx));
	}
}


void unproject(cv::Mat img, GSLAM::CameraPinhole cam)
{

	// uchar4 **ddepth, **input_image;
	// unsigned char *hdepth;
	unsigned char *d_depth_image;
	double *d_3d_points, *h_3d_points;
	int rows, cols;
	rows = img.rows;
	cols = img.cols;
	size_t totalpixels = rows*cols;
	const dim3 dimGrid((int)ceil((cols)/16), (int)ceil((rows)/16));
	const dim3 dimBlock(16, 16);
	std::cout<<"t";
	// *input_image = (uchar4 *)img.ptr<uchar4 *>(0);
	// cudaMalloc(ddepth, sizeof(uchar4) * totalpixels * CHANNELS);
	// cudaMemcpy(*ddepth, *input_image, sizeof(uchar4) * totalpixels * CHANNELS, cudaMemcpyHostToDevice);

	// h_3d_points = (double*)malloc(sizeof(double) * totalpixels * 3);
	// unsigned char* depth_image = (unsigned char*)img.data;
	// cudaMalloc((void **)&d_depth_image, sizeof(unsigned char) * totalpixels );
	// cudaMalloc((void **)&d_3d_points, sizeof(double) * totalpixels * 3 );
	// cudaMemcpy(d_depth_image, depth_image, sizeof(unsigned char) * totalpixels , cudaMemcpyHostToDevice);
	// unproject_kernel<<<dimGrid,dimBlock>>>(d_depth_image, d_3d_points, rows, cols, cam.cx, cam.cy, cam.fx, cam.fy, cam.fx_inv, cam.fy_inv);
	// cudaMemcpy(h_3d_points, d_3d_points, sizeof(double) * totalpixels * 3, cudaMemcpyDeviceToHost);
	
	// std::cout<<*(h_3d_points+sizeof(double)+(rows*cols));
	// cudaFree(d_depth_image);

	texture<double, cudaTextureType2D,  cudaReadModeElementType> t;
	cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
                                             cudaChannelFormatKindFloat );
    cudaMallocArray(&cuArray, &channelDesc, cols, rows);
    cudaMemcpyToArray(cuArray, 0, 0, img.data, sizeof(double)*totalpixels, cudaMemcpyHostToDevice);

      
}
__global__
void rgb_texture_kernel(cudaTextureObject_t tex_obj, int width, int height)
{
	int x = threadIdx.x + (blockDim.x *blockIdx.x);
	int y = threadIdx.y + (blockDim.y *blockIdx.y);

	float u = x/(float)width;
	float v = y/(float)height;

	u -= 0.5;
	v -= 0.5;

	float t = tex2D<float>(tex_obj, u,v);
	printf("%f\n",t);

}
void rgb_texture_test(cv::Mat img)
{
	int width = img.cols;
	int height = img.rows;
	int size = width * height  * sizeof(float);

	const dim3 dimGrid((int)ceil((width)/16), (int)ceil((height)/16));
	const dim3 dimBlock(16, 16);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));
	cudaSafeCall(cudaMemcpyToArray(cuArray, 0, 0, img.ptr(), size, cudaMemcpyHostToDevice));

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;    
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	rgb_texture_kernel<<<dimGrid, dimBlock>>>(texObj, width, height);

    cudaCheckError();
    cudaFreeArray(cuArray);

}