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

      
}

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef;
static cudaArray *cuArray = NULL;

__global__
void rgb_texture_kernel(unsigned char* d_img, int width, int height, int widthstep)
{
	int x = threadIdx.x + (blockDim.x *blockIdx.x);
	int y = threadIdx.y + (blockDim.y *blockIdx.y);

	  if (x >= width || y >= height)
        return;
	uchar4 t;
	t = tex2D(texRef,x,y);
	printf("x = %d y = %d  widthstep*y+x = %d\n", x, y, widthstep*y+x);
	d_img[widthstep*y+x] = t.x;
	// d_img[widthstep*(x+1)+y] = t.y;
	// d_img[widthstep*(x+2)+y] = t.z;

}
void rgb_texture_test(unsigned char* input, unsigned char* output, int width, int height, int widthstep)
{
	int nchannels = 4;

	dim3 blocksize(16,16);
	dim3 gridsize;
	gridsize.x=(width+blocksize.x-1)/blocksize.x;
	gridsize.y=(height+blocksize.y-1)/blocksize.y;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
	// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	unsigned char *d_img;
	// output = (uchar*)malloc(sizeof(uchar) * width * height * 3);
	cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));
	cudaSafeCall(cudaMemcpy2DToArray(cuArray, 0, 0, input, widthstep, width*sizeof(unsigned char), height, cudaMemcpyHostToDevice));
	cudaBindTextureToArray(texRef,cuArray,channelDesc);
	cudaSafeCall(cudaMalloc(&d_img, widthstep*height));
	// struct cudaResourceDesc resDesc;
	// memset(&resDesc, 0, sizeof(resDesc));
	// resDesc.resType = cudaResourceTypeArray;
	// resDesc.res.array.array = cuArray;

	// struct cudaTextureDesc texDesc;
	// memset(&texDesc, 0, sizeof(texDesc));
	// texDesc.addressMode[0]   = cudaAddressModeWrap;
	// texDesc.addressMode[1]   = cudaAddressModeWrap;
	// texDesc.filterMode       = cudaFilterModeLinear;
	// texDesc.readMode         = cudaReadModeElementType;    
	// texDesc.normalizedCoords = 0;

	// cudaTextureObject_t texObj = 0;
    //  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	rgb_texture_kernel<<<gridsize, blocksize>>>(d_img, width, height, widthstep/sizeof(unsigned char));
	cudaSafeCall(cudaMemcpy(output, d_img, widthstep * height, cudaMemcpyDeviceToHost));
	// cv::Mat s_img(img.size(), CV_8UC3, output);

	cudaCheckError();
    cudaFree(d_img);
    cudaFreeArray(cuArray);

}

