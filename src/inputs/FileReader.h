
#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <pangolin/utils/file_utils.h>

#include <cassert>
#include <zlib.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <stack>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class FileReader
{
    public:
        FileReader(std::string rgb_info, std::string depth_info, std::string dataset_dir, int width, int height);

        virtual ~FileReader();

        void getNext();

        void getBack();

        int getNumFrames();

        bool hasMore();

        bool rewound();

        void rewind();

        void fastForward(int frame);

        const std::string getFile();

        void setAuto(bool value);

        std::stack<int> filePointers;
        cv::Mat depth, rgb;

        int currentFrame;
        int numFrames;
        int width, height;

        int numPixels;
        FILE * fp_rgb;
        FILE * fp_depth;
        std::string dataset_dir;

};

#endif /* FILEREADER_H_ */
