#include "Segmentation.h"
// #include <time.h>
Segmentation::Segmentation(int width, int height/*, METHOD method*/)
{
	// TODO: Make customisable.
	// yolact = YolactTest(/*width, height*/);
	// this->method = method;
    Yolact yolact;
}

void Segmentation::performSegmentation(GPUTexture * rgb){
	

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    int fd = cudaops.imageResize(textPtr, target_width, target_height);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

	// clock_t tStart = clock();
	yolact.processFrame(fd);
	cudaops.cleanAllocations();
	// printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}