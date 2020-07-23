#include "RealSenseInterface.h"
#include <functional>
#include <thread>

#ifdef WITH_REALSENSE
RealSenseInterface::RealSenseInterface(int inWidth,int inHeight,int inFps)
    : width(inWidth),
    height(inHeight),
    fps(inFps),
    dev(nullptr),
    initSuccessful(true)
{
   static const std::string OPENCV_WINDOW = "Image window";

    auto list = ctx.query_devices();
    if (list.size() == 0){
        errorText = "No device connected.";
        initSuccessful = false;
        return;
    }


    rs2::device tmp_dev = list.front();
    dev = &tmp_dev;
    std::cout << dev->get_info(RS2_CAMERA_INFO_NAME) << " " << dev->get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;

    latestDepthIndex.assign(-1);
    latestRgbIndex.assign(-1);

    for (int i = 0; i < numBuffers; i++){
        uint8_t * newImage = (uint8_t *)calloc(width * height * 3,sizeof(uint8_t));
        rgbBuffers[i] = std::pair<uint8_t *,int64_t>(newImage,0);
    }

    for (int i = 0; i < numBuffers; i++){
        uint8_t * newDepth = (uint8_t *)calloc(width * height * 2,sizeof(uint8_t));
        uint8_t * newImage = (uint8_t *)calloc(width * height * 3,sizeof(uint8_t));
        frameBuffers[i] = std::pair<std::pair<uint8_t *,uint8_t *>,int64_t>(std::pair<uint8_t *,uint8_t *>(newDepth,newImage),0);
    }

    //setAutoExposure(true);
    //setAutoWhiteBalance(true);

    rgbCallback = new RGBCallback(lastRgbTime,
            latestRgbIndex,
            rgbBuffers);

    depthCallback = new DepthCallback(lastDepthTime,
            latestDepthIndex,
            latestRgbIndex,
            rgbBuffers,
            frameBuffers);
	rs2::frame_queue queue(numBuffers);
	std::thread t([&, inWidth, inHeight, inFps]() {
        /* rs2 hight level api start*/
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, inWidth, inHeight, RS2_FORMAT_RGB8,inFps);
        cfg.enable_stream(RS2_STREAM_DEPTH, inWidth, inHeight, RS2_FORMAT_Z16, inFps);

        std::cout<<"----[][][][][][][][][][]--------------------------"<<std::endl;

        //cfg.enable_stream(RS2_STREAM_POSE);
        cfg.enable_stream(RS2_STREAM_GYRO);
        cfg.enable_stream(RS2_STREAM_ACCEL);
        //cfg.enable_stream(RS2_STREAM_POSE);

        std::cout<<"[][][][][][][]------------------------------"<<std::endl;

        pipe.start(cfg);

        std::cout<<"[][][][][][][]------------------------------"<<std::endl;

        

        rs2::align align(RS2_STREAM_COLOR);
        
        

        /* rs2 hight level api end*/
        while (true)
        {
            auto frames = pipe.wait_for_frames();
	    auto aligned_frames = align.process(frames);

            if (rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL))
    {
        rs2_vector accel_sample = accel_frame.get_motion_data();
        std::cout << "Accel:" << accel_sample.x << ", " << accel_sample.y << ", " << accel_sample.z << std::endl;
        //...
    }       

         if (rs2::motion_frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO))
    {
        rs2_vector gyro_sample = gyro_frame.get_motion_data();
        std::cout << "Gyro:" << gyro_sample.x << ", " << gyro_sample.y << ", " << gyro_sample.z << std::endl;
        //...
    }

            
            
            rs2::depth_frame current_depth_frame = aligned_frames.get_depth_frame();
            rs2::video_frame current_color_frame = aligned_frames.get_color_frame();
            //auto frames = 
            rgbCallback->proccessor(current_color_frame);
            depthCallback->proccessor(current_depth_frame);
            const int w = current_color_frame.as<rs2::video_frame>().get_width();
            const int h = current_color_frame.as<rs2::video_frame>().get_height();

            //cv::imshow(OPENCV_WINDOW,image);


            
        }
    });
    t.detach();
//    dev->set_frame_callback(rs2::stream::depth,*depthCallback);
//    dev->set_frame_callback(rs2::stream::color,*rgbCallback);

//    dev->start();
}

RealSenseInterface::~RealSenseInterface()
{
    if(initSuccessful)
    {
        //dev->stop();

        for(int i = 0; i < numBuffers; i++)
        {
            free(rgbBuffers[i].first);
        }

        for(int i = 0; i < numBuffers; i++)
        {
            free(frameBuffers[i].first.first);
            free(frameBuffers[i].first.second);
        }

        delete rgbCallback;
        delete depthCallback;
    }
}

void RealSenseInterface::setAutoExposure(bool value)
{
//    dev->set_option(rs2::option::color_enable_auto_exposure,value);
//    rs2_set_option(RS2_OPTION_AUTO_EXPOSURE_MODE,)
}

void RealSenseInterface::setAutoWhiteBalance(bool value)
{
//    dev->set_option(rs2::option::color_enable_auto_white_balance,value);
}

bool RealSenseInterface::getAutoExposure()
{
//    return dev->get_option(rs2::option::color_enable_auto_exposure);
    return false;
}

bool RealSenseInterface::getAutoWhiteBalance()
{
    //return dev->get_option(rs2::option::color_enable_auto_white_balance);
    return false;
}
#else

RealSenseInterface::RealSenseInterface(int inWidth,int inHeight,int inFps)
    : width(inWidth),
    height(inHeight),
    fps(inFps),
    initSuccessful(false)
{
    errorText = "Compiled without Intel RealSense library";
}

RealSenseInterface::~RealSenseInterface()
{
}

void RealSenseInterface::setAutoExposure(bool value)
{
}

void RealSenseInterface::setAutoWhiteBalance(bool value)
{
}

bool RealSenseInterface::getAutoExposure()
{
    return false;
}

bool RealSenseInterface::getAutoWhiteBalance()
{
    return false;
}
#endif
