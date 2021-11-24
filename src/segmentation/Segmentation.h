// #include "YolactTest.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Segmentation{
public:
	// enum class METHOD {sample};
	Segmentation(int width, int height/*, METHOD method*/);
	void performSegmentation(int fd);

private:
	// METHOD method = METHOD::sample;
	// YolactTest yolact;
};