#include <cuda_runtime_api.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <Eigen/Core>
#include "cuda/utils.cuh"
#include "cuda/containers/device_array.hpp"
#include "DeformationGraph.h"

class Deformation {
 public:
  Deformation();
  // virtual ~Deformation();

  std::vector<GraphNode*>& getGraph();
  
  void sampleGraphModel(DeviceArray<float>& model_buffer, int count/**, int* g_count**/);

  class Constraint {
   public:
    Constraint(const Eigen::Vector3f& src, const Eigen::Vector3f& target, const uint64_t& srcTime, const uint64_t& targetTime,
               const bool relative, const bool pin = false)
        : src(src),
          target(target),
          srcTime(srcTime),
          targetTime(targetTime),
          relative(relative),
          pin(pin),
          srcPointPoolId(-1),
          tarPointPoolId(-1) {}

    Eigen::Vector3f src;
    Eigen::Vector3f target;
    uint64_t srcTime;
    uint64_t targetTime;
    bool relative;
    bool pin;
    int srcPointPoolId;
    int tarPointPoolId;
  };

  private:
    DeformationGraph def;

    std::vector<Eigen::Vector3f> pointPool;

    DeviceArray<float> sample_points;
  
    std::vector<std::pair<uint64_t, Eigen::Vector3f> > poseGraphPoints;
    std::vector<unsigned long long int> graphPoseTimes;
    std::vector<Eigen::Vector3f>* graphPosePoints;
  
    std::vector<Constraint> constraints;
    int lastDeformTime;

};