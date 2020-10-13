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

texture<uchar4,cudaTextureType2D,cudaReadModeNormalizedFloat> tex;

// __global__
// void rgb_texture_kernel(unsigned char* d_img, int width, int height, int widthstep)
// {
// 	int x = threadIdx.x + (blockDim.x *blockIdx.x);
// 	int y = threadIdx.y + (blockDim.y *blockIdx.y);




// 	  if (x >= width || y >= height)
//         return;
// 	uchar4 t;
// 	t = tex2D(texRef,x,y);
// 	// printf("x = %d y = %d  widthstep*y+x = %d\n", x, y, widthstep*y+x);
// 	// printf("width = %d    height = %d   widthstep = %d \n", width, height, widthstep);
// 	// d_img[widthstep*y+x] = t.x;
// 	d_img[widthstep*x+y] = tex2D(texRef, (3 * x) , y);
// 	d_img[widthstep*(x+1)+y] =  tex2D(texRef, (3 * x) + 1, y);
// 	d_img[widthstep*(x+2)+y] =  tex2D(texRef, (3 * x) + 2, y);

// }\

__global__
void test(char *img,int width,int heigth,int channels)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int offset = x + y*blockDim.x+gridDim.x;

    //If using normalization
    float u = x/(float)width;
    float v = y/(float)heigth;

    float4 pixel = tex2D(tex, x, y);

	img[(y*width+x)*channels+0] = pixel.x /4 * 255;
	img[(y*width+x)*channels+1] = pixel.y /4 * 255;
	img[(y*width+x)*channels+2] = pixel.z /4 * 255;
	img[(y*width+x)*channels+3] = 0;



}
void rgb_texture_test(cv::Mat img)
{


	int rows=img.rows;
	int cols=img.cols;
	int channels=img.channels();
	int width=cols,height=rows,size=rows*cols*channels;

	cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<uchar4>();
	cudaArray *cuArray;
	cudaMallocArray(&cuArray,&channelDesc,width,height);
	cudaMemcpyToArray(cuArray,0,0,img.data,size,cudaMemcpyHostToDevice);

	tex.addressMode[0]=cudaAddressModeWrap; 
	tex.addressMode[1]=cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;  
	tex.normalized =false;          //No normalization

	cudaBindTextureToArray(tex,cuArray,channelDesc);


	cv::Mat out=cv::Mat::zeros(width, height, CV_8UC4);
	char *dev_out=NULL;
	cudaMalloc((void**)&dev_out, size);

	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	test <<<dimGrid,dimBlock,0>>>(dev_out,width,height,channels);

	printf("working\n");


    cudaMemcpy(out.data,dev_out,size,cudaMemcpyDeviceToHost);

    cv::imshow("orignal",img);
    cv::imshow("smooth_image",out);
    cv::waitKey(0);

    cudaFree(dev_out);
    cudaFree(cuArray);
    cudaUnbindTexture(tex);
	// int nchannels = 4;

	// dim3 blocksize(16,16);
	// dim3 gridsize;
	// gridsize.x=(width+blocksize.x-1)/blocksize.x;
	// gridsize.y=(height+blocksize.y-1)/blocksize.y;

	// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
	// // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	// // unsigned char *d_img;
	// // output = (uchar*)malloc(sizeof(uchar) * width * height * 3);
	// cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));
	// cudaSafeCall(cudaMemcpyToArray(cuArray, 0, 0, input,  width*sizeof(unsigned char)*height, cudaMemcpyHostToDevice));
	// // cudaSafeCall(cudaMemcpy2DToArray(cuArray, 0, 0, input, widthstep, width*sizeof(unsigned char), height, cudaMemcpyHostToDevice));

	// cv::Mat imageOutput = cv::Mat(cv::Size(width, height), CV_8UC1);
	// uchar * d_img = imageOutput.data;
 
	// cudaMalloc((void**)&d_img, width * height * sizeof(unsigned char));


	// cudaBindTextureToArray(texRef, cuArray, channelDesc);
	// cudaSafeCall(cudaMalloc(&d_img, widthstep*height));
	// // struct cudaResourceDesc resDesc;
	// // memset(&resDesc, 0, sizeof(resDesc));
	// // resDesc.resType = cudaResourceTypeArray;
	// // resDesc.res.array.array = cuArray;

	// // struct cudaTextureDesc texDesc;
	// // memset(&texDesc, 0, sizeof(texDesc));
	// // texDesc.addressMode[0]   = cudaAddressModeWrap;
	// // texDesc.addressMode[1]   = cudaAddressModeWrap;
	// // texDesc.filterMode       = cudaFilterModeLinear;
	// // texDesc.readMode         = cudaReadModeElementType;    
	// // texDesc.normalizedCoords = 0;

	// // cudaTextureObject_t texObj = 0;
 //    //  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	// rgb_texture_kernel<<<gridsize, blocksize>>>(d_img, width, height, widthstep);
	
	// // cudaSafeCall(cudaDeviceSynchronize());
	// cudaSafeCall(cudaMemcpy(imageOutput.data, d_img, width * height, cudaMemcpyDeviceToHost));

	// // cv::Mat s_img(width, width, CV_8UC3, (void *)output);
	// cv::imwrite("src/MyImage.jpg", imageOutput);
	// printf("saved\n");

	// cudaCheckError();
 //    cudaFree(d_img);
 //    cudaFreeArray(cuArray);

}

