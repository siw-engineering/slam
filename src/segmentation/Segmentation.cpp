#include "Segmentation.h"
Segmentation::Segmentation(int width, int height/*, METHOD method*/)
{
	// TODO: Make customisable.
	// yolact = YolactTest(/*width, height*/);
	// this->method = method;
    Yolact yolact;
    maskTexture = new pangolin::GlTexture(550, 550,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
}

pangolin::GlTexture* Segmentation::performSegmentation(GPUTexture * rgb){
	

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    int fd = cudaops.imageResize(textPtr, target_width, target_height);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

	cv::Mat mask = yolact.processFrame(fd);
	cudaops.cleanAllocations();
	maskTexture->Upload(mask.data, GL_RGB, GL_UNSIGNED_BYTE);

	return maskTexture;
}