#include "Segmentation.h"
#include "Yolact.h"

Segmentation::Segmentation(int width, int height/*, METHOD method*/){
	// TODO: Make customisable.
	// yolact = YolactTest(/*width, height*/);
	// this->method = method;
}

void Segmentation::performSegmentation(int fd){
	// yolact_ = YolactSeg(/*width, height*/);
	Yolact yolact;
	// Yolact yolact();
	yolact.processFrame(fd);
	std::cout <<" performSegmentation " << std::endl;
}