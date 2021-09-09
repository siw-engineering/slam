#include "Segmentation.h"
#include "Yolact.h"

Segmentation::Segmentation(int width, int height/*, METHOD method*/){
	// TODO: Make customisable.
	// yolact = YolactTest(/*width, height*/);
	// this->method = method;
}

void Segmentation::performSegmentation(cv::Mat img){
	// yolact_ = YolactSeg(/*width, height*/);
	Yolact yolact;
	// Yolact yolact();
	yolact.processFrame(img);
	std::cout <<" performSegmentation " << std::endl;
}