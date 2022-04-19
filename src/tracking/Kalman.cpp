
#include "Kalman.h"



using namespace cv;
using namespace std;
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point3f pt,float dt,float Accel_noise_mag)
{
	deltatime = dt; //0.2

	kalman = new KalmanFilter( 6, 3, 0 ); 


	// Transition matrix
	kalman->transitionMatrix = (Mat_<float>(6, 6) << 1 		,0 		,0 		,deltatime 		,0 			,0,
													 0 		,1 		,0 		,0 				,deltatime	,0,
													 0 		,0 		,1 		,0 				,0 			,deltatime,
													 0 		,0 		,0 		,1 				,0 			,0,
													 0 		,0 		,0 		,0 				,1 			,0,
													 0 		,0 		,0 		,0 				,0 			,1);

	// init... 
	LastResult = pt;
	kalman->statePre.at<float>(0) = pt.x; // x
	kalman->statePre.at<float>(1) = pt.y; // y
	kalman->statePre.at<float>(2) = pt.z; // z


	kalman->statePre.at<float>(3) = 0;
	kalman->statePre.at<float>(4) = 0;
	kalman->statePre.at<float>(5) = 0; // z


	kalman->statePost.at<float>(0)=pt.x;
	kalman->statePost.at<float>(1)=pt.y;
	kalman->statePost.at<float>(2)=pt.z;


	setIdentity(kalman->measurementMatrix);

	kalman->processNoiseCov=(Mat_<float>(6, 6) << 
		pow(deltatime,4.0)/4.0	,0						,0						,pow(deltatime,3.0)/2.0		,0							,0,
		0						,pow(deltatime,4.0)/4.0	,0						,0							,pow(deltatime,3.0)/2.0 	,0,
		0						,0						,pow(deltatime,4.0)/4.0 ,0							,0							,pow(deltatime,3.0)/2.0,
		pow(deltatime,3.0)/2.0	,0						,0						,pow(deltatime,2.0)			,0							,0,					
		0						,pow(deltatime,3.0)/2.0	,0						,0							,pow(deltatime,2.0)			,0,
		0                       ,0						,pow(deltatime,3.0)/2.0 ,0							,0 							,pow(deltatime,2.0));


	kalman->processNoiseCov*=Accel_noise_mag;

	setIdentity(kalman->measurementNoiseCov, Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, Scalar::all(.1));

}
//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
}

//---------------------------------------------------------------------------
Point3f TKalmanFilter::GetPrediction()
{
	Mat prediction = kalman->predict();
	LastResult=Point3f(prediction.at<float>(0),prediction.at<float>(1), prediction.at<float>(2)); 
	return LastResult;
}
//---------------------------------------------------------------------------
Point3f TKalmanFilter::Update(Point3f p, bool DataCorrect)
{
	Mat measurement(3,1,CV_32FC1);
	if(!DataCorrect)
	{
		measurement.at<float>(0) = LastResult.x;  //update using prediction
		measurement.at<float>(1) = LastResult.y;
		measurement.at<float>(2) = LastResult.z;

	}
	else
	{
		measurement.at<float>(0) = p.x;  //update using measurements
		measurement.at<float>(1) = p.y;
		measurement.at<float>(2) = p.z;

	}
	// Correction
	Mat estimated = kalman->correct(measurement);
	LastResult.x=estimated.at<float>(0);   //update using measurements
	LastResult.y=estimated.at<float>(1);
	LastResult.z=estimated.at<float>(2);

	return LastResult;
}
//---------------------------------------------------------------------------