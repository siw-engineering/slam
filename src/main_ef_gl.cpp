#include "main_ef_gl.h"


int main(int argc, char const *argv[])
{

	//initialzations
	float width = 640;
	float height = 480;
	GSLAM::CameraPinhole cam_model(640, 480, 528, 528, 320, 240);
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

	pangolin::OpenGlRenderState s_cam;
	pangolin::CreateWindowAndBind("Main",2*width, 2*height);
	glEnable(GL_DEPTH_TEST);
    s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(2*width, 2*height, 277, 277, 2*width / 2.0f, 2*height / 2.0f, 0.1, 1000),
                                        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));
	pangolin::View& d_cam = pangolin::CreateDisplay()
	     .SetBounds(0.0, 1.0, 0.0, 1.0, -2*width*height)
	     .SetHandler(new pangolin::Handler3D(s_cam));

	RGBDOdometry frameToModel(width, height, intr.cx,intr.cy, intr.fx, intr.fy);
	RGBDOdometry modelToModel(width, height, intr.cx,intr.cy, intr.fx, intr.fy);
    Eigen::Matrix4f currPose = Eigen::Matrix4f::Identity();
    float confidence = 10.0f;
    float depth = 3.0f;
    float icp = 10.0f;
    float icpErrThresh = 5e-05;
    float covThresh = 1e-05;
    float photoThresh = 115;
    float fernThresh = 0.3095f;
    int timeDelta = 200;
    int icpCountThresh = 40000;
 	
 	// Fern
    bool closeLoops = true;
    Resolution::getInstance(640, 480);
    Intrinsics::getInstance(528, 528, 320, 240);
    Ferns ferns(500, 3 * 1000, photoThresh);
    int fernDeforms = 0;
    bool pyramid = true;
    int deforms;
    bool fastOdom = false;
    Resize resize(Resolution::getInstance().width(),
              Resolution::getInstance().height(),
              Resolution::getInstance().width() / 20,
              Resolution::getInstance().height() / 20);


    Img<Eigen::Vector4f> consBuff(Resolution::getInstance().rows() / 20, Resolution::getInstance().cols() / 20);
    Img<unsigned short> timesBuff(Resolution::getInstance().rows() / 20, Resolution::getInstance().cols() / 20);


    Deformation localDeformation;
    Deformation globalDeformation;
    std::vector<PoseMatch> poseMatches;
    std::vector<Deformation::Constraint> relativeCons;

    std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > poseGraph;
    std::vector<unsigned long long int> poseLogTimes;


    // resize TO DO
    float depthCutoff = 3;
    float maxDepthProcessed = 20;
 	std::map<std::string, GPUTexture*> textures;
	std::map<std::string, ComputePack*> computePacks;
	std::map<std::string, FeedbackBuffer*> feedbackBuffers;

	//createtextures
	textures[GPUTexture::RGB] = new GPUTexture(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
    textures[GPUTexture::DEPTH_RAW] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
    textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT, false, true);
    textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(width, height, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    textures[GPUTexture::DEPTH_NORM] = new GPUTexture(width, height, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT, true);

    //createcompute
    computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"), textures[GPUTexture::DEPTH_NORM]->texture, width, height);
    computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"), textures[GPUTexture::DEPTH_FILTERED]->texture, width, height);
    computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"), textures[GPUTexture::DEPTH_METRIC]->texture, width, height);
    computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"), textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, width, height);

    //createfeedbackbuffers
    feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"), width, height, intr);
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"), width, height, intr);
	

	LogReader * logReader;
  	std::string logFile;
  	logFile = "/home/developer/work/ElasticFusion/dyson_lab.klg";
    logReader = new RawLogReader(logFile, false, width, height);
    IndexMap indexMap(width, height, intr);
    GlobalModel globalModel(width, height, intr);
    FillIn fillIn(width, height, intr);


    std::shared_ptr<Shader> draw_program;
    draw_program =  std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface_.vert","draw_global_surface_.frag"));

    int tick = 1;
    int64_t timestamp;

	while (logReader->hasMore())
	{
		logReader->getNext();
		timestamp = logReader->timestamp;
		textures[GPUTexture::DEPTH_RAW]->texture->Upload(logReader->depth, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
		textures[GPUTexture::RGB]->texture->Upload(logReader->rgb, GL_RGB, GL_UNSIGNED_BYTE);

		std::vector<Uniform> uniformsfd;
		uniformsfd.push_back(Uniform("cols", (float)width));
		uniformsfd.push_back(Uniform("rows", (float)height));
		uniformsfd.push_back(Uniform("maxD", depthCutoff));
		computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniformsfd);

		std::vector<Uniform> uniforms;
		uniforms.push_back(Uniform("maxD", depthCutoff));
		computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
		computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);

		if (tick == 1)
		{
			feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC]->texture, tick, maxDepthProcessed);
   			feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, tick, maxDepthProcessed);

	        globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);
	        frameToModel.initFirstRGB(textures[GPUTexture::RGB]);	
		}
		else
		{
       		Eigen::Matrix4f lastPose = currPose;
       		bool trackingOk = frameToModel.lastICPError < 1e-04;
            bool shouldFillIn = true;
            frameToModel.initICPModel(shouldFillIn ? &fillIn.vertexTexture : indexMap.vertexTex(),
                                      shouldFillIn ? &fillIn.normalTexture : indexMap.normalTex(),
                                      maxDepthProcessed, currPose);
            frameToModel.initRGBModel((shouldFillIn || false) ? &fillIn.imageTexture : indexMap.imageTex());
            frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
            frameToModel.initRGB(textures[GPUTexture::RGB]);
            Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);
            frameToModel.getIncrementalTransformation(trans, rot, false, 0.3, true, false, true);
            currPose.topRightCorner(3, 1) = trans;
            currPose.topLeftCorner(3, 3) = rot;

			Eigen::Matrix4f diff = currPose.inverse() * lastPose;
			Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
			Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

			//Weight by velocity
			float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());
			float largest = 0.01;
			float minWeight = 0.5;

			if(weighting > largest)
				weighting = largest;

			weighting = std::max(1.0f - (weighting / largest), minWeight) * 1;
	   		predict(indexMap, currPose, globalModel, maxDepthProcessed, confidence, tick, timeDelta, fillIn, textures);
			Eigen::Matrix4f recoveryPose = currPose;
			std::vector<Ferns::SurfaceConstraint> constraints;

            if(closeLoops){
	            std::cout << "Ferns::findFrame" << std::endl;
	            recoveryPose = ferns.findFrame(constraints,
	                                           currPose,
	                                           &fillIn.vertexTexture,
	                                           &fillIn.normalTexture,
	                                           &fillIn.imageTexture,
	                                           tick,
	                                           false);
	            std::cout << "Ferns::findFrame" << std::endl;

            }

        	std::vector<float> rawGraph;
            bool fernAccepted = false;
            if (ferns.lastClosest != -1 && closeLoops ){

                for(size_t i = 0; i < constraints.size(); i++)
                {
                    globalDeformation.addConstraint(constraints.at(i).sourcePoint,
                                                    constraints.at(i).targetPoint,
                                                    tick,
                                                    ferns.frames.at(ferns.lastClosest)->srcTime,
                                                    true);
                }
            	std::cout << " loop closure detected -------------" << std::endl;

                for(size_t i = 0; i < relativeCons.size(); i++)
                {
                    globalDeformation.addConstraint(relativeCons.at(i));
                }

                std::cout << " globalDeformation calling -------------" << std::endl;
                if(globalDeformation.constrain(ferns.frames, rawGraph, tick, true, poseGraph, true))
                {
                    currPose = recoveryPose;

                    poseMatches.push_back(PoseMatch(ferns.lastClosest, ferns.frames.size(), ferns.frames.at(ferns.lastClosest)->pose, currPose, constraints, true));

                    fernDeforms += rawGraph.size() > 0;

                    fernAccepted = true;
                }

            }

            if(closeLoops && rawGraph.size()==0)
            {
            	indexMap.combinedPredict(currPose,
            							 globalModel.model(),
            							 maxDepthProcessed,
            							 confidence,
            							 0,
            							 tick - timeDelta,
            							 timeDelta,
            							 IndexMap::INACTIVE);

            //WARNING initICP* must be called before initRGB*
            modelToModel.initICPModel(indexMap.oldVertexTex(), indexMap.oldNormalTex(), maxDepthProcessed, currPose);
            modelToModel.initRGBModel(indexMap.oldImageTex());

            modelToModel.initICP(indexMap.vertexTex(), indexMap.normalTex(), maxDepthProcessed);
            modelToModel.initRGB(indexMap.imageTex());

            Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

            modelToModel.getIncrementalTransformation(trans,
                                                      rot,
                                                      false,
                                                      10,
                                                      pyramid,
                                                      fastOdom,
                                                      false);

            Eigen::MatrixXd covar = modelToModel.getCovariance();
            bool covOk = true;

            for(int i = 0; i < 6; i++)
            {
                if(covar(i, i) > covThresh)
                {
                    covOk = false;
                    break;
                }
            }

            Eigen::Matrix4f estPose = Eigen::Matrix4f::Identity();

            estPose.topRightCorner(3, 1) = trans;
            estPose.topLeftCorner(3, 3) = rot;



            if(covOk && modelToModel.lastICPCount > icpCountThresh && modelToModel.lastICPError < icpErrThresh)
            {
                resize.vertex(indexMap.vertexTex(), consBuff);
                resize.time(indexMap.oldTimeTex(), timesBuff);

                for(int i = 0; i < consBuff.cols; i++)
                {
                    for(int j = 0; j < consBuff.rows; j++)
                    {
                        if(consBuff.at<Eigen::Vector4f>(j, i)(2) > 0 &&
                           consBuff.at<Eigen::Vector4f>(j, i)(2) < maxDepthProcessed &&
                           timesBuff.at<unsigned short>(j, i) > 0)
                        {
                            Eigen::Vector4f worldRawPoint = currPose * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0),
                                                                                       consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                                                       consBuff.at<Eigen::Vector4f>(j, i)(2),
                                                                                       1.0f);

                            Eigen::Vector4f worldModelPoint = estPose * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0),
                                                                                        consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                                                        consBuff.at<Eigen::Vector4f>(j, i)(2),
                                                                                        1.0f);
                            constraints.push_back(Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));
                            // std::cout << " localDeformation adding constrains -----" << std::endl;

                            localDeformation.addConstraint(worldRawPoint,
                                                           worldModelPoint,
                                                           tick,
                                                           timesBuff.at<unsigned short>(j, i),
                                                           deforms == 0);
                        }
                    }
                }

                std::vector<Deformation::Constraint> newRelativeCons;
                std::cout << " ----------------localDeformation constrains --------------------" << std::endl;
                if(localDeformation.constrain(ferns.frames, rawGraph, tick, false, poseGraph, false, &newRelativeCons))
                {

                    std::cout << " localDeformation detected ---------------------------------------- " << std::endl;
                    poseMatches.push_back(PoseMatch(ferns.frames.size() - 1, ferns.frames.size(), estPose, currPose, constraints, false));

                    deforms += rawGraph.size() > 0;

                    currPose = estPose;

                    for(size_t i = 0; i < newRelativeCons.size(); i += newRelativeCons.size() / 3)
                    {
                        relativeCons.push_back(newRelativeCons.at(i));
                    }

                }
            }

            }

			if (trackingOk)
	        {
	            indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);
	            globalModel.fuse(currPose,
	                             tick,
	                             textures[GPUTexture::RGB],
	                             textures[GPUTexture::DEPTH_METRIC],
	                             textures[GPUTexture::DEPTH_METRIC_FILTERED],
	                             indexMap.indexTex(),
	                             indexMap.vertConfTex(),
	                             indexMap.colorTimeTex(),
	                             indexMap.normalRadTex(),
	                             maxDepthProcessed,
	                             confidence,
	                             weighting);

	            indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);




	            if(rawGraph.size() > 0 && !fernAccepted)
	            {
	                std::cout << " synthesizeDepth " << std::endl;
	                indexMap.synthesizeDepth(currPose,
	                                         globalModel.model(),
	                                         maxDepthProcessed,
	                                         confidence,
	                                         tick,
	                                         tick - timeDelta,
	                                         std::numeric_limits<unsigned short>::max());
	            }


	            globalModel.clean(currPose,
	                              tick,
	                              indexMap.indexTex(),
	                              indexMap.vertConfTex(),
	                              indexMap.colorTimeTex(),
	                              indexMap.normalRadTex(),
	                              indexMap.depthTex(),
	                              confidence,
	                              rawGraph,
	                              timeDelta,
	                              maxDepthProcessed,
	                              false);
	        }
        }


	    poseGraph.push_back(std::pair<unsigned long long int, Eigen::Matrix4f>(tick, currPose));
	    poseLogTimes.push_back(timestamp);
	    TICK("sampleGraph");

	    localDeformation.sampleGraphModel(globalModel.model());

	    globalDeformation.sampleGraphFrom(localDeformation);


	    predict(indexMap, currPose, globalModel, maxDepthProcessed, confidence, tick, timeDelta, fillIn, textures);
	    std::cout << "tick : " << tick << std::endl;
		ferns.addFrame(&fillIn.imageTexture, &fillIn.vertexTexture, &fillIn.normalTexture, currPose, tick, fernThresh);
		tick++;

		visualize_mb(draw_program, GetMvp(s_cam), globalModel.model(), Vertex::SIZE);
        pangolin::FinishFrame();

	}
	return 0;
}