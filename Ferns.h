#ifndef FERNS_H_
#define FERNS_H_
#include <cuda_runtime_api.h>

#include <iostream>
#include <random>
#include <Eigen/Core>
#include <vector>
#include <limits>
#include "Img.h"
#include "cuda/cudafuncs.cuh"
#include "cuda/containers/device_array.hpp"
#include "cuda/utils.cuh"
#include "RGBDOdometry.h"


class Ferns {
 public:
  Ferns(int n, int maxDepth, const float photoThreshconst, int resolution_w,
          const int resolution_h, const CameraModel& intr);
  virtual ~Ferns();

  bool addFrame(DeviceArray<float>& imageTexture, DeviceArray2D<float>& vertexTexture, DeviceArray2D<float>& normalTexture,
          const Eigen::Matrix4f& pose, int srcTime, const float threshold);

  class SurfaceConstraint {
   public:
    SurfaceConstraint(const Eigen::Vector4f& sourcePoint, const Eigen::Vector4f& targetPoint)
        : sourcePoint(sourcePoint), targetPoint(targetPoint) {}

    Eigen::Vector4f sourcePoint;
    Eigen::Vector4f targetPoint;
  };

  Eigen::Matrix4f findFrame(std::vector<SurfaceConstraint>& constraints, const Eigen::Matrix4f& currPose, DeviceArray2D<float>& vertexTexture,
                            DeviceArray2D<float>& normalTexture, DeviceArray<float>& imageTexture, const int time, const bool lost);

  class Fern {
   public:
    Fern() {}

    Eigen::Vector2i pos;
    Eigen::Vector4i rgbd;
    std::vector<int> ids[16];
  };

  std::vector<Fern> conservatory;

  class Frame {
   public:
    Frame(int n, int id, const Eigen::Matrix4f& pose, const int srcTime, const int numPixels, unsigned char* rgb = 0,
          Eigen::Vector4f* verts = 0, Eigen::Vector4f* norms = 0)
        : goodCodes(0), id(id), pose(pose), srcTime(srcTime), initRgb(rgb), initVerts(verts), initNorms(norms) {
      codes = new unsigned char[n];

      if (rgb) {
        this->initRgb = new unsigned char[numPixels * 3];
        memcpy(this->initRgb, rgb, numPixels * 3);
      }

      if (verts) {
        this->initVerts = new Eigen::Vector4f[numPixels];
        memcpy(this->initVerts, verts, numPixels * sizeof(Eigen::Vector4f));
      }

      if (norms) {
        this->initNorms = new Eigen::Vector4f[numPixels];
        memcpy(this->initNorms, norms, numPixels * sizeof(Eigen::Vector4f));
      }
    }

    virtual ~Frame() {
      delete[] codes;

      if (initRgb) delete[] initRgb;

      if (initVerts) delete[] initVerts;

      if (initNorms) delete[] initNorms;
    }

    unsigned char* codes;
    int goodCodes;
    const int id;
    Eigen::Matrix4f pose;
    const int srcTime;
    unsigned char* initRgb;
    Eigen::Vector4f* initVerts;
    Eigen::Vector4f* initNorms;
  };

  std::vector<Frame*> frames;

  const int num;
  std::mt19937 random;
  const int factor;
  const int maxDepth;
  const float photoThresh;
  const int width;
  const int height;
  const int intr_cx;
  const int intr_cy;
  const float intr_fx;
  const float intr_fy;
  std::uniform_int_distribution<int32_t> widthDist;
  std::uniform_int_distribution<int32_t> heightDist;
  std::uniform_int_distribution<int32_t> rgbDist;
  std::uniform_int_distribution<int32_t> dDist;



  int lastClosest;
  const unsigned char badCode;
  RGBDOdometry rgbd;

 private:
  void generateFerns();

  float blockHD(const Frame* f1, const Frame* f2);
  float blockHDAware(const Frame* f1, const Frame* f2);

  float photometricCheck(const Img<Eigen::Vector4f>& vertSmall, const Img<Eigen::Matrix<unsigned char, 3, 1>>& imgSmall,
                         const Eigen::Matrix4f& estPose, const Eigen::Matrix4f& fernPose, const unsigned char* fernRgb);

  DeviceArray2D<float> vertFern;
  DeviceArray<float> vertCurrent;

  DeviceArray2D<float> normFern;
  DeviceArray<float> normCurrent;

  // DeviceArray<float> colorFern;
  // DeviceArray<float> colorCurrent;


  Img<Eigen::Matrix<unsigned char, 3, 1>> imageBuff;
  Img<Eigen::Vector4f> vertBuff;
  Img<Eigen::Vector4f> normBuff;
};

#endif /* FERNS_H_ */
