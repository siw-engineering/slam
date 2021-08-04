#include "FileReader.h"

FileReader::FileReader(std::string rgb_info, std::string depth_info, std::string dataset_dir, int width, int height)
:dataset_dir(dataset_dir),
width(width),
height(height)
{
    assert(pangolin::FileExists(rgb_info.c_str()));
    assert(pangolin::FileExists(depth_info.c_str()));
    char ch;
    numFrames = 0;
    currentFrame = 0;
    fp_rgb = fopen(rgb_info.c_str(), "r");
    while (EOF != (ch=getc(fp_rgb)))
        if ('\n' == ch)
            ++numFrames;

    numFrames = numFrames - 3;
    fclose(fp_rgb);

    fp_rgb = fopen(rgb_info.c_str(), "r");
    fp_depth = fopen(depth_info.c_str(), "r");

    char tmp[100];
    fscanf(fp_rgb,"%s %s %s", tmp, tmp, tmp);
    fscanf(fp_rgb,"%s %s %s", tmp, tmp, tmp);
    fscanf(fp_rgb,"%s %s %s", tmp, tmp, tmp);

    fscanf(fp_depth,"%s %s %s", tmp, tmp, tmp);
    fscanf(fp_depth,"%s %s %s", tmp, tmp, tmp);
    fscanf(fp_depth,"%s %s %s", tmp, tmp, tmp);

}

FileReader::~FileReader()
{
    fclose(fp_rgb);
    fclose(fp_depth);

}

void FileReader::getNext()
{
    if (hasMore())
    {
        char frame[100], file_rgb[200], file_depth[200];
        fscanf(fp_rgb,"%s %s ", frame, file_rgb);
        fscanf(fp_depth,"%s %s ", frame, file_depth);
        rgb = cv::imread(dataset_dir+file_rgb, CV_LOAD_IMAGE_ANYCOLOR);
        depth = cv::imread(dataset_dir+file_depth, CV_LOAD_IMAGE_ANYDEPTH);
        currentFrame++;
    }

}

int FileReader::getNumFrames()
{
    return numFrames;
}

bool FileReader::hasMore()
{
    return currentFrame + 1 < numFrames;
}


