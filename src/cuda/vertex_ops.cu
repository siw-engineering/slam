#include "vertex_ops.cuh"

__global__ void unproject_kernel(void)
{
	printf("fuckyea\n");
}


void unproject(cv::Mat img, GSLAM::CameraPinhole cam)
{

	int thread = 16;

	uchar4 **drgb, **input_image;
	unsigned char *hrgb;
	int CHANNELS;
	int rows, cols;

	rows = img.rows;
	cols = img.cols;
	CHANNELS = 3;
	size_t totalpixels = rows*cols;


	*input_image = (uchar4 *)img.ptr<unsigned char>(0);

	// cudaMalloc(drgb, sizeof(uchar4) * totalpixels * CHANNELS);
	// cudaMemcpy(*drgb, *input_image, sizeof(uchar4) * totalpixels * CHANNELS, cudaMemcpyHostToDevice);

	unproject_kernel<<<1,1>>>();

	// cudaFree(drgb);
}