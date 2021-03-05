#include "FillIn.h"

FillIn::FillIn(int width, int height)
	: width(width), height(height)
{
  imageTexture.create(height*4*width);
  vertexTexture.create(height, width);
  normalTexture.create(height, width);

}

FillIn::~FillIn() {}

void FillIn::image(DeviceArray<float>& existingRgb, DeviceArray<float>& rawRgb, bool passthrough){

  fillinRgb(width, height, existingRgb, rawRgb, passthrough, imageTexture);

}

void FillIn::vertex(const CameraModel& intr, DeviceArray2D<float>& existingVertex,
	DeviceArray2D<float>& rawDepth, bool passthrough)
{
	fillinVertex(intr, width, height, existingVertex, rawDepth, passthrough, vertexTexture);
}

void FillIn::normal(const CameraModel& intr, DeviceArray2D<float>& existingNormal, DeviceArray2D<float>& rawDepth, bool passthrough)
{
	fillinNormal(intr, width, height, existingNormal, rawDepth, passthrough, normalTexture);
}