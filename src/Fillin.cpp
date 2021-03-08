#include "FillIn.h"

FillIn::FillIn(int width, int height)
	: width(width), height(height)
{

}

FillIn::~FillIn() {}

DeviceArray<float> FillIn::image(DeviceArray<float>& existingRgb, DeviceArray<float>& rawRgb, bool passthrough){

  DeviceArray<float> imageTexture;
  imageTexture.create(height*4*width);
  fillinRgb(width, height, existingRgb, rawRgb, passthrough, imageTexture);
  return imageTexture;
}

DeviceArray2D<float> FillIn::vertex(const CameraModel& intr, DeviceArray2D<float>& existingVertex,
	DeviceArray2D<float>& rawDepth, bool passthrough)
{
	DeviceArray2D<float> vertexTexture;
  	vertexTexture.create(height, width);
	fillinVertex(intr, width, height, existingVertex, rawDepth, passthrough, vertexTexture);
	return vertexTexture;
}

DeviceArray2D<float> FillIn::normal(const CameraModel& intr, DeviceArray2D<float>& existingNormal, DeviceArray2D<float>& rawDepth, bool passthrough)
{
  	DeviceArray2D<float> normalTexture;
	normalTexture.create(height, width);
	fillinNormal(intr, width, height, existingNormal, rawDepth, passthrough, normalTexture);
	return normalTexture;
}