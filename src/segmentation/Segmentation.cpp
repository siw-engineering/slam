#include "Segmentation.h"
Segmentation::Segmentation(int width, int height/*, METHOD method*/)
{
	// TODO: Make customisable.
	// yolact = YolactTest(/*width, height*/);
	// this->method = method;
    Yolact yolact;
    // RGB mask
    // maskTexture = new pangolin::GlTexture(550, 550,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
    // 1 channel mask
    textures[GPUTexture::MASK] = new GPUTexture(width, height, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, false, true);
}

pangolin::GlTexture* Segmentation::performSegmentation(GPUTexture * rgb){
	

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    int fd = cudaops.imageResize(textPtr, target_width, target_height);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

	cv::Mat mask = yolact.processFrame(fd);
	cudaops.cleanAllocations();
	// maskTexture->Upload(mask.data, GL_RGB, GL_UNSIGNED_BYTE);
	textures[GPUTexture::MASK]->texture->Upload(mask.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);

	return textures[GPUTexture::MASK]->texture;
}


void Segmentation::getBBox(GPUTexture * rgb, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr,  int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height)
{

	cudaArray * textPtr;
    cudaGraphicsMapResources(1, &rgb->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);
    int fd = cudaops.imageResize(textPtr, target_width, target_height);
    cudaGraphicsUnmapResources(1, &rgb->cudaRes);
    yolact.computeBBox(fd, bbox_verts_ptr, bbox_ele_ptr, no, depth, cx, cy, fx, fy, width, height);
  
}