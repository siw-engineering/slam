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

using namespace std;


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

int main(int argc, char  *argv[])
{


	int width, height, rows, cols;
	width = cols = 320;
	height = rows = 240;
	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity(), lastpose = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f tinv;
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

	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;
	DepthSubscriber* depthsub;
	RGBSubscriber* rgbsub;
	cv::Mat dimg, img;
	RGBDOdometry* rgbd_odom;

	GSLAM::CameraPinhole cam_model(320,240,277,277,160,120);
	CameraModel intr;
	intr.cx = cam_model.cx;
	intr.cy = cam_model.cy;
	intr.fx = cam_model.fx;
	intr.fy = cam_model.fy;

	rgbd_odom = new RGBDOdometry(width, height, (float)cam_model.cx, (float)cam_model.cy, (float)cam_model.fx, (float)cam_model.fy);

	DeviceArray<float> rgb, rgb_prev, color_splat, vmaps_tmp, nmaps_tmp;
	DeviceArray2D<float> depth;
	DeviceArray2D<float> vmap, nmap, vmap_splat_prev, nmap_splat_prev/*, color_splat*/;
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

	depthsub  = new DepthSubscriber("/X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);


	// float* dval = new float[width*height*3];

	FillIn fillin(width, height);
	DeviceArray<float> fillin_img;
	DeviceArray2D<float> fillin_vt, fillin_nt;

	fillin_vt.create(height*4, width);
	fillin_nt.create(height*4, width);
	fillin_img.create(height*4*width);

	int frame = 0;
	while (ros::ok())
	{
		img  = rgbsub->read();
		dimg = depthsub->read();
		if (dimg.empty() || img.empty())
		{
			ros::spinOnce();
			continue;	
		}
		img.convertTo(img, CV_32FC3);
		dimg.convertTo(dimg, CV_32FC1);
		rgb.upload((float*)img.data, height*3*width);
		depth.upload((float*)dimg.data, width*sizeof(float), height, width);
		/*debug on*/

			// void* data_ = 0;
			// int sizeBytes_ = width*height*3*sizeof(float);

			//upload
			// cudaSafeCall(cudaMalloc(&data_, sizeBytes_));
			// cudaSafeCall(cudaMemcpy(data_, (float*)img.data, sizeBytes_, cudaMemcpyHostToDevice));
			// cudaSafeCall(cudaDeviceSynchronize());

			//download
			// cudaSafeCall(cudaMemcpy(r, data_, sizeBytes_, cudaMemcpyDeviceToHost));
			// cudaSafeCall(cudaDeviceSynchronize());
			// float* r = new float[width*height*3];
			// unsigned char* out = new unsigned char[width*height];
			// rgb.download(r);
			// int jj = 0;
			// for (int ii =0; ii<width*height; ii++)
			// {
			// 		out[ii] = (unsigned char)r[jj];
			// 		jj+=3;
			// }
			// cv::Mat save_img(rows, cols, CV_8UC1, out);
			// cv::imwrite("src/test1.jpg", save_img);
			// exit(0);
		/*off*/
		if (frame==0)
		{
			rgbd_odom->initFirstRGB(rgb);
			createVMap(intr, depth, vmap, depthCutOff);
			createNMap(vmap, nmap);
			rgb_prev.upload((float*)img.data, height*3*width);
			tinv  = pose.inverse();
			initModelBuffer(intr, depthCutOff, model_buffer, &count, vmap, nmap, rgb);
			splatDepthPredict(intr, height, width,  maxDepth, tinv.data(), model_buffer, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
			ros::spinOnce();
			frame++;
			continue;
		}

		//generateCUDATextures
		copyDMaps(depth, depthPyr[0]);
		for (int i = 1; i < 3; ++i) 
			pyrDownGaussF(depthPyr[i - 1], depthPyr[i]);
		cudaDeviceSynchronize();
		cudaCheckError();
		tinv  = pose.inverse();

		// perfrom tracking
		fillin.vertex(intr, vmap_splat_prev, depth, fillin_vt, true);
		fillin.normal(intr, nmap_splat_prev, depth, fillin_nt, true);
		fillin.image(color_splat, rgb, fillin_img, true);
		
		rgbd_odom->initICPModel(fillin_vt, fillin_nt, maxDepth, pose);
		copyMaps(fillin_vt, fillin_nt, vmaps_tmp, nmaps_tmp);
		rgbd_odom->initRGBModel(fillin_img, vmaps_tmp);
	

		// rgbd_odom->initICP(depthPyr, maxDepth);
		// rgbd_odom->initRGB(rgb, vmaps_tmp);
		// transObject = pose.topRightCorner(3, 1);
		// rotObject = pose.topLeftCorner(3, 3);
		// rgbd_odom->getIncrementalTransformation(transObject, rotObject, false, 0.3, true, false, true, 0, 0);
		
		// pose.topRightCorner(3, 1) = transObject;
		// pose.topLeftCorner(3, 3) = rotObject;

		// //predict indicies
		// predictIndicies(intr, rows, cols, maxDepth, tinv.data(), model_buffer, i/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count);

		// //fuse
		// float w = computeFusionWeight(1, pose.inverse()*lastpose);
		// fuse(depth, rgb, intr, rows, cols, maxDepth, pose.data(), model_buffer, i/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count, w);


		// std::cout<<"i :"<< i<< "\ntrans :"<<transObject<<std::endl<<"rot :"<<rotObject<<std::endl;

		// //splat predict
		// createVMap(intr, depth, vmap, 100);
		// createNMap(vmap, nmap);
		// std::swap(rgb, rgb_prev);
		// splatDepthPredict(intr, height, width,  maxDepth, tinv.data(), model_buffer, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);

		frame++;			// TO DO set lastpose
		ros::spinOnce();
	}

	return 0;
}