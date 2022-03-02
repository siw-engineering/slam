// #include "YolactTest.h"
#include "../gl/types.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.h"
#include "Yolact.h"


class Segmentation{
public:
	// enum class METHOD {sample};
	Segmentation(int width, int height/*, METHOD method*/);
	pangolin::GlTexture* performSegmentation(GPUTexture * rgb);


	int target_width = 550;
	int target_height = 550;

private:
	// METHOD method = METHOD::sample;
	Yolact yolact;
	CudaOps cudaops;
	std::map<std::string, GPUTexture*> textures;
	// pangolin::GlTexture* maskTexture;
};