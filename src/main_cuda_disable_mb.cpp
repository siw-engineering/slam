#include <cuda_runtime_api.h>
#include "inputs/ros/DepthSubscriber.h"
#include "inputs/ros/RGBSubscriber.h"
#include "inputs/ros/PoseSubscriber.h"
#include "Camera.h"
#include <iostream>
#include "cuda/cudafuncs.cuh"
#include "cuda/utils.cuh"
#include "cuda/containers/device_array.hpp"
#include "RGBDOdometry.h"
#include <unistd.h>
#include "Render.h"
#include <cstdlib>

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
	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity(), lastpose = Eigen::Matrix4f::Identity(), drawpose =Eigen::Matrix4f::Identity();
	Eigen::Matrix4f tinv;
	int count = 0;
	float depthCutOff, maxDepth;
	maxDepth = depthCutOff = 5;
	float confThreshold = 9;
	int timeDelta = 200;

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
	PoseSubscriber* posesub;
	cv::Mat dimg, img;
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
    DeviceArray2D<unsigned int> time_splat;
	DeviceArray2D<unsigned int> index_pi;
	DeviceArray2D<float> neighbours_and_vert;
	

	DeviceArray<float> model_buffer, model_buffer_rs;
	DeviceArray2D<float> updateVConf, updateNormRad, updateColTime, unstable_buffer;
	std::vector<DeviceArray2D<float>> depthPyr;


    float* vertices_splat = new float[rows*cols*4];
    memset(&vertices_splat[0], 0, rows*cols*4);

    vmap_splat_prev.create(rows*4, cols); // TO DO put it outside
    vmap_splat_prev.upload(&vertices_splat[0], sizeof(float)*cols, 4*rows, cols);

    color_splat.create(rows*4*cols);
    color_splat.upload(&vertices_splat[0], rows*4*cols);

    nmap_splat_prev.create(rows*4, cols);
    nmap_splat_prev.upload(&vertices_splat[0], sizeof(float)*cols, 4*rows, cols);
    
    time_splat.create(rows,cols);
    time_splat.upload(&vertices_splat[0], sizeof(float)*cols, rows, cols);

    delete[] vertices_splat;	

    float* init_nv = new float[17*3];
    memset(&init_nv[0], 0, 17*3);

    neighbours_and_vert.create(17,3);
    neighbours_and_vert.upload(&init_nv[0], sizeof(float)*3, 17, 3);

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

	// model_buffer.create(bufferSize);
	// model_buffer_rs.create(bufferSize);
	// unstable_buffer.create(width, height*VSIZE*3);
	// updateVConf.create(TEXTURE_DIMENSION, TEXTURE_DIMENSION*4);
	// updateNormRad.create(TEXTURE_DIMENSION, TEXTURE_DIMENSION*4);
	// updateColTime.create(TEXTURE_DIMENSION, TEXTURE_DIMENSION*4);


	// initialize model buffers
	float* vertices_mb = new float[bufferSize];
	memset(&vertices_mb[0], 0, bufferSize);
	model_buffer.upload(&vertices_mb[0], bufferSize);
	model_buffer_rs.upload(&vertices_mb[0], bufferSize);
	// delete[] vertices_mb;
	
	float* vertices = new float[TEXTURE_DIMENSION*TEXTURE_DIMENSION*4];
	memset(&vertices[0], 0, TEXTURE_DIMENSION*TEXTURE_DIMENSION*4);
	
	float* ub_vertices = new float[width*height*VSIZE];
	memset(&ub_vertices[0], 0, width*height*VSIZE);



	depthsub  = new DepthSubscriber("/X1/front/depth", nh);
	rgbsub = new RGBSubscriber("/X1/front/image_raw", nh);
	posesub = new PoseSubscriber("/X1/odom", nh);


	// float* dval = new float[width*height*3];

	// FillIn fillin(width, height);
	DeviceArray<float> fillin_img;
	DeviceArray2D<float> fillin_vt, fillin_nt;

	fillin_vt.create(height*4, width);
	fillin_nt.create(height*4, width);
	fillin_img.create(height*4*width);

	int frame = 0;

    int up, usp, cvw0, cvwm1;
    up = 0;
    usp = 0;
    cvw0 = 0;
    cvwm1 = 0;
	int psize;
	int update_count = 0;

    //debug
 //    int ib_len = 1000;
	// DeviceArray<float> imagebin;
	// imagebin.create(width*height*ib_len);
	// int ibcount = 0;

	// float* imgzeros = new float[width*height*ib_len];
	// memset(&imgzeros[0], 0, width*height*ib_len);
	// imagebin.upload(&imgzeros[0], width*height*ib_len);
	// delete[] imgzeros;
	//off
	pose = posesub->read();




	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = pose.topLeftCorner(3, 3);
	Eigen::Vector3f tcam = pose.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam_inv;
	Eigen::Vector3f tcam_inv;


	mat33 device_Rcam = Rcam;
	float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());
	mat33 device_Rcam_inv;
	float3 device_tcam_inv;




	int windowsize = 8;
	int window_multiplier = 2;
	int windowarea = pow(2*windowsize*window_multiplier,2);

	DeviceArray2D<float> outframe, piframe;
	outframe.create(3*2*windowsize*window_multiplier, 2*windowsize*window_multiplier);
	piframe.create(3*2*windowsize*window_multiplier, 2*windowsize*window_multiplier);
	DeviceArray<float> vmap_mb;
//render.h stuff
	Render view(640, 480);
	int objects;
	objects = 2;
	int oattrib[objects*4];

	int point_count[objects];
	// point_count[0] = windowarea;
	// point_count[1] = windowarea;

	int obj_rgb[3*objects];

	// obj_rgb[0] = 255;
	// obj_rgb[1] = 255;	
	// obj_rgb[2] = 255;	

	// obj_rgb[3] = 255;	
	// obj_rgb[4] = 0;	
	// obj_rgb[5] = 0;	

	// // obj_rgb[6] = 0;	
	// // obj_rgb[7] = 255;	
	// // obj_rgb[8] = 0;	

	// for (int i = 0; i < objects; ++i)
	// {
	// 	oattrib[i*4] = point_count[i];
	// 	oattrib[i*4 + 1] = obj_rgb[i*3];
	// 	oattrib[i*4 + 2] = obj_rgb[i*3 + 1];
	// 	oattrib[i*4 + 3] = obj_rgb[i*3 + 2];

	// }
	// view.setObjects(objects, oattrib);

	int pc=0;
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
        computeBilateralFilter(depth, depthf, depthCutOff);


		if (frame==0)
		{
			createVMap(intr, depth, vmap, depthCutOff);
			createNMap(vmap, nmap);
			rgbd_odom->initFirstRGB(rgb);
			model_buffer.upload(&vertices_mb[0], bufferSize);
			initModelBuffer(intr, depthCutOff, model_buffer, device_Rcam, device_tcam, &count, vmap, nmap, rgb);
			
			tinv  = pose.inverse();
			Rcam_inv = tinv.topLeftCorner(3,3);
			tcam_inv = tinv.topRightCorner(3,1);
			device_Rcam_inv = Rcam_inv;
			device_tcam_inv = *reinterpret_cast<float3*>(tcam_inv.data());

			splatDepthPredict(intr, height, width, model_buffer, maxDepth, confThreshold, frame, frame, timeDelta, device_Rcam_inv, device_tcam_inv, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
			fillinVertex(intr, width, height, vmap_splat_prev, depth, false, fillin_vt);
			fillinNormal(intr, width, height, nmap_splat_prev, depth, false, fillin_nt);
			fillinRgb(width, height, color_splat, rgb, false, fillin_img);
			ros::spinOnce();
			frame++;
			continue;
		}

		// generateCUDATextures
		copyDMaps(depth, depthPyr[0]);
		for (int i = 1; i < 3; ++i) 
			pyrDownGaussF(depthPyr[i - 1], depthPyr[i]);
		// cudaDeviceSynchronize();
		// cudaCheckError();

		// rgbd_odom->initICPModel(fillin_vt, fillin_nt, maxDepth, pose);
		// copyMaps(fillin_vt, fillin_nt, vmaps_tmp, nmaps_tmp);
		// rgbd_odom->initRGBModel(fillin_img, vmaps_tmp);
		// rgbd_odom->initICP(depthPyr, maxDepth);
		// rgbd_odom->initRGB(rgb, vmaps_tmp);

		// transObject = pose.topRightCorner(3, 1);
		// rotObject = pose.topLeftCorner(3, 3);
		// rgbd_odom->getIncrementalTransformation(transObject, rotObject, false, 0.3, true, false, true, 0, 0);
		// if(!(isnan(transObject[0])))
		// 	pose.topRightCorner(3, 1) = transObject;
		// pose.topLeftCorner(3, 3) = rotObject;

		pose = posesub->read();
		Rcam = pose.topLeftCorner(3, 3);
		tcam = pose.topRightCorner(3, 1);
		device_Rcam = Rcam;
		device_tcam = *reinterpret_cast<float3*>(tcam.data());

		tinv  = pose.inverse();
		Rcam_inv = tinv.topLeftCorner(3,3);
		tcam_inv = tinv.topRightCorner(3,1);
		device_Rcam_inv = Rcam_inv;
		device_tcam_inv = *reinterpret_cast<float3*>(tcam_inv.data());

		// splatDepthPredict(intr, height, width, model_buffer, maxDepth, confThreshold, frame, frame, timeDelta, device_Rcam_inv, device_tcam_inv, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
		// fillinVertex(intr, width, height, vmap_splat_prev, depth, false, fillin_vt);
		// fillinNormal(intr, width, height, nmap_splat_prev, depth, false, fillin_nt);
		// fillinRgb(width, height, color_splat, rgb, false, fillin_img);

		updateVConf.upload(vertices, TEXTURE_DIMENSION*sizeof(float), TEXTURE_DIMENSION*4, TEXTURE_DIMENSION);
		updateNormRad.upload(vertices, TEXTURE_DIMENSION*sizeof(float), TEXTURE_DIMENSION*4, TEXTURE_DIMENSION);
		updateColTime.upload(vertices, TEXTURE_DIMENSION*sizeof(float), TEXTURE_DIMENSION*4, TEXTURE_DIMENSION);
		unstable_buffer.upload(ub_vertices, width*sizeof(float), height*4, width);
 
		predictIndicies(&pc, intr, rows, cols, maxDepth, device_Rcam_inv, device_tcam_inv, model_buffer, frame/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count);
		float w = computeFusionWeight(1, pose.inverse()*lastpose);
		fuse_data(&up, &usp, depth, rgb, depthf, intr, rows, cols, maxDepth, device_Rcam, device_tcam, model_buffer, frame, vmap_pi, ct_pi, nmap_pi, index_pi, w, updateVConf, updateNormRad, updateColTime, unstable_buffer); // predict indices
		fuse_update(&cvw0, &cvwm1, intr, rows, cols, maxDepth, device_Rcam, device_tcam, model_buffer, model_buffer_rs, frame, &count, updateVConf, updateNormRad, updateColTime);       // predict indices
		predictIndicies(&pc, intr, rows, cols, maxDepth, device_Rcam_inv, device_tcam_inv, model_buffer_rs, frame/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count);
		clean(depthf, intr, rows, cols, maxDepth, device_Rcam_inv, device_tcam_inv, model_buffer, model_buffer_rs, frame, timeDelta, confThreshold, &count, vmap_pi, ct_pi, nmap_pi, index_pi, updateVConf, updateNormRad, updateColTime, unstable_buffer);

		// splatDepthPredict(intr, height, width,  maxDepth, tinv.data(), model_buffer, count, color_splat, vmap_splat_prev, nmap_splat_prev, time_splat);
		// fillinVertex(intr, width, height, vmap_splat_prev, depth, false, fillin_vt);
		// fillinNormal(intr, width, height, nmap_splat_prev, depth, false, fillin_nt);
		// fillinRgb(width, height, color_splat, rgb, false, fillin_img);

		// std::cout<<"udpate points :"<<up<<" unstable points :"<<usp<<" count:"<<count<<std::endl;
		// std::cout<<"cvw0 :"<<cvw0<<" cvwm1 :"<<cvwm1<<std::endl;


		//debug on
			createVMap(intr, depth, vmap, depthCutOff);
			// createNMap(vmap, nmap);
			// float w = computeFusionWeight(1, pose.inverse()*lastpose);
			// predictIndicies(&pc, intr, rows, cols, maxDepth, device_Rcam_inv, device_tcam_inv, model_buffer, frame/*time*/, vmap_pi, ct_pi, nmap_pi, index_pi, count);
			// // normalFusion(model_buffer, &count, depth, intr, rows, cols, maxDepth, pose.data());
			// normalFusionData(model_buffer, &count, &update_count, frame, depth, intr, rows, cols, maxDepth, device_Rcam, device_tcam, w, vmap_pi, ct_pi, nmap_pi, index_pi, neighbours_and_vert);
			// // exp(depth, vmap_pi, outframe, piframe, intr, rows, cols, maxDepth);
			// std::cout<<"update_count :"<<update_count<<" count :"<<count<<std::endl;

		// debug on
			//vmap_pi download
				// float* vmap_pi_hst = new float[height*width*4];
				// vmap_pi.download(vmap_pi_hst, width*sizeof(float));
				
			//modelbuffer download
				extractVmap(model_buffer, count, vmap_mb, device_Rcam_inv, device_tcam_inv);
				float* vmap_mb_hst = new float[count*3];
				vmap_mb.download(vmap_mb_hst);

			// // //vmap download
				float* vmap_hst = new float[height*width*3];
				vmap.download(vmap_hst, width*sizeof(float));

			// float* points = new float[2*height*width*3];
			// std::copy(vmap_pi_hst, vmap_pi_hst+(height*width), points);
			// std::copy(vmap_hst, vmap_hst+(height*width), points+(height*width));
			
			// std::copy(vmap_pi_hst+(height*width), vmap_pi_hst+(2*height*width), points+2*(height*width));
			// std::copy(vmap_hst+(height*width), vmap_hst+(2*height*width), points+3*(height*width));

			// std::copy(vmap_pi_hst+(2*height*width), vmap_pi_hst+(3*height*width), points+4*(height*width));
			// std::copy(vmap_hst+(2*height*width), vmap_hst+(3*height*width), points+5*(height*width));

			// // std::copy(mb_xxx, mb_xxx+(height*width*3), points+(height*width*3));
			// // std::copy(vmap_hst, vmap_hst+(height*width*3), points+(height*width*3));


			// delete[] vmap_pi_hst;
			// delete[] vmap_hst;

			float plot[3*width*height+3*count];
			view.xxxtoxyz(vmap_hst, plot, width*height);
			std::copy(vmap_mb_hst, vmap_mb_hst+count*3, plot+(height*width*3));

			objects = 2;
			point_count[0] = height*width;
			point_count[1] = count;

			obj_rgb[0] = 255;
			obj_rgb[1] = 255;	
			obj_rgb[2] = 255;	

			obj_rgb[3] = 0;
			obj_rgb[4] = 255;	
			obj_rgb[5] = 0;	

			for (int i = 0; i < objects; ++i)
			{
				oattrib[i*4] = point_count[i];
				oattrib[i*4 + 1] = obj_rgb[i*3];
				oattrib[i*4 + 2] = obj_rgb[i*3 + 1];
				oattrib[i*4 + 3] = obj_rgb[i*3 + 2];

			}
			view.setObjects(objects, oattrib);
			view.bufferHandle(plot, sizeof(plot));
			view.draw("vertex.vert", "draw.frag", GL_POINTS);

		// off

		//debug on
			// float* outframe_hst = new float[windowarea*3];
			// outframe.download(outframe_hst, 2*windowsize*window_multiplier*sizeof(float));

			// float* piframe_hst = new float[windowarea*3];
			// piframe.download(piframe_hst, 2*windowsize*window_multiplier*sizeof(float));
			
			// float* points = new float[2*windowarea*3];

			// std::copy(outframe_hst, outframe_hst+(windowarea), points);
			// std::copy(piframe_hst, piframe_hst+(windowarea), points+(windowarea));

			// std::copy(outframe_hst+(windowarea), outframe_hst+(2*windowarea), points+2*(windowarea));
			// std::copy(piframe_hst+(windowarea), piframe_hst+(2*windowarea), points+3*(windowarea));

			// std::copy(outframe_hst+(2*windowarea), outframe_hst+(3*windowarea), points+4*(windowarea));
			// std::copy(piframe_hst+(2*windowarea), piframe_hst+(3*windowarea), points+5*(windowarea));

			// float* plot = new float[2*windowarea*3];
			// view.xxxtoxyz(points, plot, 2*windowarea);

			// // for (int i = 0; i < windowarea; ++i)
			// // {
			// // 	std::cout<<points[i*3]<<" "<<points[i*3+1]<<" "<<points[i*3+2]<<std::endl;
			// // }
			// view.bufferHandle(plot, 2*windowarea*3*sizeof(float));
			// view.draw("vertex.vert", "draw.frag", GL_POINTS);

		//debug on
			// float* r = new float[width*height*4];
			// unsigned char* out = new unsigned char[width*height];
			// fillin_img.download(r);
			// int jj = 0;
			// for (int ii =0; ii<width*height; ii++)
			// {	
			// 		out[ii] = (unsigned char)r[jj];
			// 		jj+=4;
			// }
			// cv::Mat save_img(rows, cols, CV_8UC1, out);
			// cv::imwrite("src/testr.jpg", save_img);
		//off


		// float* mb = new float[bufferSize];
		// model_buffer.download(mb);
		// // if (count < 588716)
		// 	psize = count;
		// float* mb_xyz = new float[psize*3];
		// std::copy(mb, mb+psize, mb_xyz);
		// std::copy(mb+(3072*3072), mb+(3072*3072)+psize, mb_xyz+psize);
		// std::copy(mb+2*(3072*3072), mb+2*(3072*3072)+psize, mb_xyz+(2*psize));
		// // std::copy(mb+(8*3072*3072), mb+(8*3072*3072)+(height*width), mb_xyz);
		// // std::copy(mb+(9*3072*3072), mb+(9*3072*3072)+(height*width), mb_xyz+(height*width));
		// // std::copy(mb+(10*3072*3072), mb+(10*3072*3072)+(height*width), mb_xyz+(2*height*width));
		// delete[] mb;
		// float plot[psize*3];
		// view.glCoord(mb_xyz, plot, psize);
		// view.bufferHandle(plot, sizeof(plot));
		// view.draw("vertex.vert", "draw.frag", GL_POINTS, psize);

		// std::cout<<"frame :"<<frame<<" update points:"<<up<<" unstable points:"<<usp<<std::endl;
		// std::cout<<"frame :"<<frame<<" cvw0 points:"<<cvw0<<" cvwm1 points:"<<cvwm1<<std::endl;
		// std::cout<<"pose:\n"<<pose<<std::endl;
		// std::cout<<"count :"<<count<<std::endl;
		// if (frame > 0)
			// break;
		// std::cout<< "\ntrans :"<<transObject<<std::endl<<"rot :"<<rotObject<<std::endl;


		// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// drawpose.topRightCorner(3, 1) = transObject;
		// drawpose.topLeftCorner(3, 3) = rotObject;
		// glLineWidth(4);
		// pangolin::glDrawFrustum(Kinv, 640, 480, pose, 0.2f);
		// glLineWidth(1);
		
		// std::cout<<pose<<std::endl;
		// std::cout<<"pc :"<<pc<<std::endl;
		// std::cout<<"count :"<<count<<std::endl;


		pc = 0;
		up = 0;
		usp = 0;
		cvw0 = 0;
  		cvwm1 = 0;
		lastpose = pose;
		frame++;		
		update_count = 0;
		// count = 0;	// TO DO set lastpose
		ros::spinOnce();
		pangolin::FinishFrame();

	}

    // debug

	return 0;
}	