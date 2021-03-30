#include <cuda_runtime_api.h>
#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"
#include "Camera.h"
#include <iostream>
#include "cuda/cudafuncs.cuh"
#include "cuda/containers/device_array.hpp"
#include "RGBDOdometry.h"
#include "FillIn.h"
#include <unistd.h>
#include "Render.h"
// #include "GPUTexture.h"

#include <librealsense2/rs.hpp> 

using namespace GSLAM;

Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix) 
{
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

  double rx = R(2, 1) - R(1, 2);
  double ry = R(0, 2) - R(2, 0);
  double rz = R(1, 0) - R(0, 1);

  double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
  double c = (R.trace() - 1) * 0.5;
  c = c > 1. ? 1. : c < -1. ? -1. : c;

  double theta = acos(c);

  if (s < 1e-5) {
    double t;

    if (c > 0)
      rx = ry = rz = 0;
    else {
      t = (R(0, 0) + 1) * 0.5;
      rx = sqrt(std::max(t, 0.0));
      t = (R(1, 1) + 1) * 0.5;
      ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
      t = (R(2, 2) + 1) * 0.5;
      rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

      if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry * rz > 0)) rz = -rz;
      theta /= sqrt(rx * rx + ry * ry + rz * rz);
      rx *= theta;
      ry *= theta;
      rz *= theta;
    }
  } else {
    double vth = 1 / (2 * s);
    vth *= theta;
    rx *= vth;
    ry *= vth;
    rz *= vth;
  }
  return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

float computeFusionWeight(float weightMultiplier, Eigen::Matrix4f diff) 
{

  Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
  Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

  float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());

  const float largest = 0.01;
  const float minWeight = 0.5;

  if (weighting > largest) weighting = largest;

  weighting = std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

  return weighting;
}



int main(int argc, char **argv)
{

    int width, height, rows, cols;
    width = cols = 320;
    height = rows = 240;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity(), lastpose = Eigen::Matrix4f::Identity(), drawpose =Eigen::Matrix4f::Identity();  Eigen::Matrix4f tinv;
    int count = 0;
    float depthCutOff, maxDepth;
    maxDepth = depthCutOff = 5;


    const int TEXTURE_DIMENSION = 3072;
    const int MAX_VERTICES = TEXTURE_DIMENSION * TEXTURE_DIMENSION;
    const int NODE_TEXTURE_DIMENSION = 16384;
    const int MAX_NODES = NODE_TEXTURE_DIMENSION / 16;  // 16 floats per node
    // int VSIZE = sizeof(Eigen::Vector4f) ;
    int VSIZE = 4;
    const int bufferSize = MAX_VERTICES * VSIZE * 3;
    // int d2d_w, d2d_h;
    // d2d_w = sqrt(MAX_VERTICES);
    // d2d_h = d2d_w;

    int weightMultiplier = 1;

    Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);


    int inWidth = 640, inHeight = 480, inFps = 30;
    rs2::pipeline pipe;
    rs2::config cfg;
    //cfg.enable_device_from_file("/home/sathish/Documents/20201113_111309.bag");
    cfg.enable_stream(RS2_STREAM_COLOR, inWidth, inHeight, RS2_FORMAT_RGB8,inFps);
    cfg.enable_stream(RS2_STREAM_DEPTH, inWidth, inHeight, RS2_FORMAT_Z16, inFps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, inWidth, inHeight, RS2_FORMAT_Y8, inFps);
    pipe.start(cfg);

    rs2::align align(RS2_STREAM_INFRARED);
    GSLAM::CameraPinhole cam(320,240,277,277,160,120);

    // cv::Mat dimg, img;
    RGBDOdometry* rgbd_odom;

    GSLAM::CameraPinhole cam_model(320,240,277,277,160,120);
    CameraModel intr;
    intr.cx = cam_model.cx;
    intr.cy = cam_model.cy;
    intr.fx = cam_model.fx;
    intr.fy = cam_model.fy;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = cam_model.fx;
    K(1, 1) = cam_model.fy;
    K(0, 2) = cam_model.cx;
    K(1, 2) = cam_model.cy;
    Eigen::Matrix3f Kinv = K.inverse();

    rgbd_odom = new RGBDOdometry(width, height, (float)cam_model.cx, (float)cam_model.cy, (float)cam_model.fx, (float)cam_model.fy);

    DeviceArray<float> rgb, rgb_prev, color_splat, vmaps_tmp, nmaps_tmp;
    DeviceArray2D<float> depth, depthf;
    DeviceArray2D<float> vmap, nmap, vmap_splat_prev, nmap_splat_prev/*, color_splat*/, vmap_test;
    DeviceArray2D<unsigned char> lastNextImage;
    DeviceArray2D<float> vmap_pi, nmap_pi, ct_pi;
    DeviceArray2D<unsigned int> index_pi;

    DeviceArray2D<unsigned int> time_splat;
    DeviceArray<float> model_buffer, unstable_buffer;
    std::vector<DeviceArray2D<float>> depthPyr;

    depthPyr.resize(3);
    for (int i = 0; i < 3; ++i) 
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;
        depthPyr[i].create(pyr_rows, pyr_cols);
    }

    rgb.create(height*3*width);
    rgb_prev.create(height*3*width);
    depth.create(height, width);
    lastNextImage.create(height, width);

    model_buffer.create(bufferSize);
    unstable_buffer.create(width*height*VSIZE);

    // initialize model buffers
    float* vertices = new float[bufferSize];
    memset(&vertices[0], 0, bufferSize);
    model_buffer.upload(&vertices[0], bufferSize);
    delete[] vertices;

    FillIn fillin(width, height);
    DeviceArray<float> fillin_img;
    DeviceArray2D<float> fillin_vt, fillin_nt;

    fillin_vt.create(height*4, width);
    fillin_nt.create(height*4, width);
    fillin_img.create(height*4*width);

    Render view(640, 480);

    int frame = 0;


    while (true)
    {

        auto frames = pipe.wait_for_frames();
        auto aligned_frames = align.process(frames);

        rs2::depth_frame current_depth_frame = aligned_frames.get_depth_frame();
        rs2::video_frame current_color_frame = aligned_frames.get_color_frame();

        std::cout << " test" << std::endl;
        Mat img(Size(inWidth, inHeight), CV_8UC3, (void*)current_color_frame.get_data() , Mat::AUTO_STEP);
        Mat dimg(Size(inWidth, inHeight), CV_16UC1, (void*)current_depth_frame.get_data() , Mat::AUTO_STEP);

        cv::resize(img, img, Size(width, height));
        cv::resize(dimg, dimg, Size(width, height));

        // cv::imshow("dimag", img);
        // cv::waitKey(0);
        img.convertTo(img, CV_32FC3);
        dimg.convertTo(dimg, CV_32FC1);
        // cv::cvtColor(img, img, CV_BGR2RGB);

        /*
        cv::cvtColor(image, image, CV_BGR2RGB);


        Mat hsv, color_image, dark_image;

        cv::cvtColor(image,hsv,CV_BGR2HSV);

        const auto result = cv::mean(hsv);

        if (result[2]<128)
        {   
            rs2::video_frame current_infrared_frame = aligned_frames.get_infrared_frame(1);
            Mat infrared_image(Size(640, 480), CV_8UC1, (void*)current_infrared_frame.get_data() , Mat::AUTO_STEP);
            //imshow("INFRARED Image", infrared_image);
            img = infrared_image.clone();

        }
        else{

            img = image.clone();

        }

        cv::cvtColor(img, img, CV_BGR2BGRA);
        */

        rgb.upload((float*)img.data, height*3*width);
        depth.upload((float*)dimg.data, width*sizeof(float), height, width);
        computeBilateralFilter(depth, depthf, depthCutOff);


        if (frame==0)
        {
            createVMap(intr, depth, vmap, depthCutOff);
            createNMap(vmap, nmap);
            rgbd_odom->initFirstRGB(rgb);
            rgb_prev.upload((float*)img.data, height*3*width);
            tinv  = pose.inverse();
            initModelBuffer(intr, depthCutOff, model_buffer, &count, vmap, nmap, rgb);
            splatDepthPredict(intr, height, width,  maxDepth, tinv.data(), model_buffer, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
            fillin.vertex(intr, vmap_splat_prev, depth, fillin_vt, false);
            fillin.normal(intr, nmap_splat_prev, depth, fillin_nt, false);
            fillin.image(color_splat, rgb, fillin_img, false);
            frame++;
            continue;
        }

        copyDMaps(depth, depthPyr[0]);
        for (int i = 1; i < 3; ++i) 
            pyrDownGaussF(depthPyr[i - 1], depthPyr[i]);
        cudaDeviceSynchronize();
        cudaCheckError();


        rgbd_odom->initICPModel(fillin_vt, fillin_nt, maxDepth, pose);
        copyMaps(fillin_vt, fillin_nt, vmaps_tmp, nmaps_tmp);
        rgbd_odom->initRGBModel(fillin_img, vmaps_tmp);
        rgbd_odom->initICP(depthPyr, maxDepth);
        rgbd_odom->initRGB(rgb, vmaps_tmp);

        transObject = pose.topRightCorner(3, 1);
        rotObject = pose.topLeftCorner(3, 3);
        rgbd_odom->getIncrementalTransformation(transObject, rotObject, false, 0.3, true, false, true, 0, 0);
        pose.topRightCorner(3, 1) = transObject;
        pose.topLeftCorner(3, 3) = rotObject;

        //predict()
        tinv  = pose.inverse();
        splatDepthPredict(intr, height, width,  maxDepth, tinv.data(), model_buffer, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
    


        fillin.vertex(intr, vmap_splat_prev, depth, fillin_vt, false);
        fillin.normal(intr, nmap_splat_prev, depth, fillin_nt, false);
        fillin.image(color_splat, rgb, fillin_img, false);


        predictIndicies(intr, rows, cols, maxDepth, tinv.data(), model_buffer, frame/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count);

        float w = computeFusionWeight(1, pose.inverse()*lastpose);
        fuse(depth, rgb, depthf, intr, rows, cols, maxDepth, pose.data(), model_buffer, &count, frame, vmap_pi, ct_pi, nmap_pi, index_pi, w);       // predict indices
        predictIndicies(intr, rows, cols, maxDepth, tinv.data(), model_buffer, frame/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count);

        // splat predict
        splatDepthPredict(intr, height, width,  maxDepth, tinv.data(), model_buffer, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
        fillin.vertex(intr, vmap_splat_prev, depth, fillin_vt, false);
        fillin.normal(intr, nmap_splat_prev, depth, fillin_nt, false);
        fillin.image(color_splat, rgb, fillin_img, false);

        drawpose.topLeftCorner(3, 3) = rotObject;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLineWidth(4);
        pangolin::glDrawFrustum(Kinv, 640, 480, pose, 0.2f);
        glLineWidth(1);


        std::cout<< "\ntrans :"<<transObject<<std::endl<<"rot :"<<rotObject<<std::endl;
        lastpose = pose;
        frame++;
        pangolin::FinishFrame();

    }

    return 0;
}