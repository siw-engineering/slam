// #ifndef FILLIN_H_
// #define FILLIN_H_
#include <cuda_runtime_api.h>
#include "cuda/utils.cuh"
#include "cuda/containers/device_array.hpp"



class FillIn {
 public:
  FillIn(int width, int height);
  virtual ~FillIn();
  DeviceArray<float> image(DeviceArray<float>& existingRgb, DeviceArray<float>& rawRgb, bool passthrough);

  DeviceArray2D<float> vertex(const CameraModel& intr, DeviceArray2D<float>& existingVertex,
              DeviceArray2D<float>& rawDepth, bool passthrough);

  DeviceArray2D<float> normal(const CameraModel& intr, DeviceArray2D<float>& existingNormal, DeviceArray2D<float>& rawDepth,
              bool passthrough);

  int width;
  int height;


  // std::shared_ptr<Shader> imageProgram;
  // pangolin::GlRenderBuffer imageRenderBuffer;
  // pangolin::GlFramebuffer imageFrameBuffer;

  // std::shared_ptr<Shader> vertexProgram;
  // pangolin::GlRenderBuffer vertexRenderBuffer;
  // pangolin::GlFramebuffer vertexFrameBuffer;

  // std::shared_ptr<Shader> normalProgram;
  // pangolin::GlRenderBuffer normalRenderBuffer;
  // pangolin::GlFramebuffer normalFrameBuffer;

};
// #endif /* FILLIN_H_ */
