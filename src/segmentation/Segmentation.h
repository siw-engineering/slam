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
    void getBBox(GPUTexture * rgb, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr,  int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height);

	int target_width = 550;
	int target_height = 550;


private:
	// METHOD method = METHOD::sample;
	Yolact yolact;
	CudaOps cudaops;
	std::map<std::string, GPUTexture*> textures;
	// pangolin::GlTexture* maskTexture;
};