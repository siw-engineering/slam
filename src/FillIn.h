// #ifndef FILLIN_H_
// #define FILLIN_H_
#include <cuda_runtime_api.h>
#include "cuda/utils.cuh"
#include "cuda/containers/device_array.hpp"



class FillIn {
 public:
  FillIn(int width, int height);
  virtual ~FillIn();
  void image(DeviceArray<float>& existingRgb, DeviceArray<float>& rawRgb, bool passthrough);

  void vertex(const CameraModel& intr, DeviceArray2D<float>& existingVertex,
              DeviceArray2D<float>& rawDepth, bool passthrough);

  void normal(const CameraModel& intr, DeviceArray2D<float>& existingNormal, DeviceArray2D<float>& rawDepth,
              bool passthrough);

  DeviceArray<float> imageTexture;
  DeviceArray2D<float> vertexTexture;
  DeviceArray2D<float> normalTexture;
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
