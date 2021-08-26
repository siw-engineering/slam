#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

void printImgStats(cv::Mat img)
{
	double minVal; 
	double maxVal; 
	std::string cvtype;

	minMaxLoc(img, &minVal, &maxVal);

	std::cout<<"img.type :"<<img.type()<<"("<<getImageType(img.type())<<")"<<std::endl;
	std::cout<<"min val  :"<<minVal<<std::endl;
	std::cout<<"max val  :"<<maxVal<<std::endl;
}
