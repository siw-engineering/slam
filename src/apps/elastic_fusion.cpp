#include "inputs/RawLogReader.h"
#include <libconfig.hh>
#include "../ui/EFGUI.h"
// #include "../odom/RGBDOdometryef.h"
#include "../gl/ComputePack.h"
#include "../gl/FillIn.h"
#include "../model/GlobalModel.h"
#include "../lc/Deformation.h"
#include "../lc/PoseMatch.h"


using namespace libconfig;


void predict(IndexMap& indexMap, Eigen::Matrix4f& currPose, GlobalModel& globalModel, int maxDepthProcessed, float confidenceThreshold, int tick, int timeDelta, FillIn& fillIn, std::map<std::string, GPUTexture*>& textures)
{
    if(/*lastFrameRecovery*/false)
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 0,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    else
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 tick,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], false);
    fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], false);
    fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], false || /*frameToFrameRGB*/false);
}

Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}


void savePly(GlobalModel& globalModel, std::string saveFilename, float confidenceThreshold)
{
    std::string filename = saveFilename;
    filename.append(".ply");

    // Open file
    std::ofstream fs;
    fs.open (filename.c_str ());

    Eigen::Vector4f * mapData = globalModel.downloadMap();

    int validCount = 0;

    for(unsigned int i = 0; i < globalModel.lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 3) + 0];

        if(pos[3] > confidenceThreshold)
        {
            validCount++;
        }
    }

    // Write header
    fs << "ply";
    fs << "\nformat " << "binary_little_endian" << " 1.0";

    // Vertices
    fs << "\nelement vertex "<< validCount;
    fs << "\nproperty float x"
          "\nproperty float y"
          "\nproperty float z";

    fs << "\nproperty uchar red"
          "\nproperty uchar green"
          "\nproperty uchar blue";

    fs << "\nproperty float nx"
          "\nproperty float ny"
          "\nproperty float nz";

    fs << "\nproperty float radius";

    fs << "\nend_header\n";

    // Close the file
    fs.close ();

    // Open file in binary appendable
    std::ofstream fpout (filename.c_str (), std::ios::app | std::ios::binary);

    for(unsigned int i = 0; i < globalModel.lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 3) + 0];

        if(pos[3] > confidenceThreshold)
        {
            Eigen::Vector4f col = mapData[(i * 3) + 1];
            Eigen::Vector4f nor = mapData[(i * 3) + 2];

            nor[0] *= -1;
            nor[1] *= -1;
            nor[2] *= -1;

            float value;
            memcpy (&value, &pos[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            unsigned char r = int(col[0]) >> 16 & 0xFF;
            unsigned char g = int(col[0]) >> 8 & 0xFF;
            unsigned char b = int(col[0]) & 0xFF;

            fpout.write (reinterpret_cast<const char*> (&r), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&g), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&b), sizeof (unsigned char));

            memcpy (&value, &nor[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[3], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));
        }
    }

    // Close file
    fs.close ();

    delete [] mapData;
}


int main(int argc, char const *argv[])
{

    Config cfg;
    try
    {
        cfg.readFile("/home/developer/slam/src/configs/ef_iclnuim.cfg");
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return(EXIT_FAILURE);
    }

    // ElasticFusion Params
    float confidence, icpWeight, icpErrThresh, covThresh, photoThresh, fernThresh, depthCutoff, maxDepthProcessed;
    int timeDelta, icpCountThresh, deforms, fernDeforms;
    bool closeLoops, pyramid, fastOdom, rgbOnly;
    const Setting& root = cfg.getRoot();
    root["ef"].lookupValue("confidence", confidence);
    root["ef"].lookupValue("icpWeight", icpWeight);
    root["ef"].lookupValue("icpErrThresh", icpErrThresh);
    root["ef"].lookupValue("covThresh", covThresh);
    root["ef"].lookupValue("photoThresh", photoThresh);
    root["ef"].lookupValue("fernThresh", fernThresh);
    root["ef"].lookupValue("timeDelta", timeDelta);
    root["ef"].lookupValue("icpCountThresh", icpCountThresh);
    root["ef"].lookupValue("depthCutoff", depthCutoff);
    root["ef"].lookupValue("maxDepthProcessed", maxDepthProcessed);
    root["ef"].lookupValue("closeLoops", closeLoops);
    root["ef"].lookupValue("pyramid", pyramid);
    root["ef"].lookupValue("fastOdom", fastOdom);
    root["ef"].lookupValue("rgbOnly", rgbOnly);

    //Camera Params
    CameraModel intr(0,0,0,0,0,0);
    root["camera"].lookupValue("width", intr.width);
    root["camera"].lookupValue("height", intr.height);
    root["camera"].lookupValue("fx", intr.fx);
    root["camera"].lookupValue("fy", intr.fy);
    root["camera"].lookupValue("cx", intr.cx);
    root["camera"].lookupValue("cy", intr.cy);

    //GUI
    float width, height;
    root["gui"].lookupValue("width", width);
    root["gui"].lookupValue("height", height);

    //Shaders
    std::string gl_shaders, model_shaders, lc_shaders, ui_shaders;
    root["shaders"].lookupValue("gl", gl_shaders);
    root["shaders"].lookupValue("model", model_shaders);
    root["shaders"].lookupValue("lc", lc_shaders);
    root["shaders"].lookupValue("ui", ui_shaders);

    bool sply;
    std::string saveply_file;
    root["saveply"].lookupValue("save", sply);
    root["saveply"].lookupValue("file", saveply_file);

    EFGUI gui(width, height, intr.cx, intr.cy, intr.fx, intr.fy, ui_shaders);
    RGBDOdometryef frameToModel(width, height, intr.cx,intr.cy, intr.fx, intr.fy);

    // LC
    RGBDOdometryef modelToModel(width, height, intr.cx,intr.cy, intr.fx, intr.fy);
    Ferns ferns(500, depthCutoff * 1000, photoThresh, intr, width, height, gl_shaders);

    Deformation localDeformation(lc_shaders);
    Deformation globalDeformation(lc_shaders);

    std::vector<PoseMatch> poseMatches;
    std::vector<Deformation::Constraint> relativeCons;

    std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > poseGraph;
    std::vector<unsigned long long int> poseLogTimes;
    Resize resize(width, height, width / 20, height / 20, gl_shaders);

    Img<Eigen::Vector4f> consBuff(height / 20, width / 20);
    Img<unsigned short> timesBuff(height / 20, width / 20);


    //data
    std::string logFile;
    root["data"].lookupValue("path", logFile);
    LogReader * logReader;
    logReader = new RawLogReader(logFile, false, width, height);

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
    computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_NORM]->texture, width, height);
    computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_FILTERED]->texture, width, height);
    computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_METRIC]->texture, width, height);
    computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom", gl_shaders), textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, width, height);

    //createfeedbackbuffers
    feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom", gl_shaders), width, height, intr);
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom", gl_shaders), width, height, intr);
    

    IndexMap indexMap(width, height, intr, model_shaders);
    GlobalModel globalModel(width, height, intr, model_shaders);
    FillIn fillIn(width, height, intr, gl_shaders);


    int tick = 1;
    int64_t timestamp;
    fernDeforms = 0;
    Eigen::Matrix4f currPose = Eigen::Matrix4f::Identity();

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
            frameToModel.getIncrementalTransformation(trans, rot, rgbOnly, icpWeight, pyramid, fastOdom, true);
            currPose.topRightCorner(3, 1) = trans;
            currPose.topLeftCorner(3, 3) = rot;

            Eigen::Matrix4f diff = currPose.inverse() * lastPose;
            Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
            Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

            //Weight by velocity
            float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());
            float largest = 0.01;
            float minWeight = 0.5;
            std::vector<float> rawGraph;

            if(weighting > largest)
                weighting = largest;

            weighting = std::max(1.0f - (weighting / largest), minWeight) * 1;
            predict(indexMap, currPose, globalModel, maxDepthProcessed, confidence, tick, timeDelta, fillIn, textures);
            Eigen::Matrix4f recoveryPose = currPose;


            std::vector<Ferns::SurfaceConstraint> constraints;

            if (closeLoops){

                recoveryPose = ferns.findFrame(constraints,
                                               currPose,
                                               &fillIn.vertexTexture,
                                               &fillIn.normalTexture,
                                               &fillIn.imageTexture,
                                               tick,
                                               false);
            }

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
                // std::cout << " loop closure detected -------------" << std::endl;

                for(size_t i = 0; i < relativeCons.size(); i++)
                {
                    globalDeformation.addConstraint(relativeCons.at(i));
                }

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

                                localDeformation.addConstraint(worldRawPoint,
                                                               worldModelPoint,
                                                               tick,
                                                               timesBuff.at<unsigned short>(j, i),
                                                               deforms == 0);
                            }
                        }
                    }

                    std::vector<Deformation::Constraint> newRelativeCons;
                    if(localDeformation.constrain(ferns.frames, rawGraph, tick, false, poseGraph, false, &newRelativeCons))
                    {

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
        localDeformation.sampleGraphModel(globalModel.model());
        globalDeformation.sampleGraphFrom(localDeformation);

        predict(indexMap, currPose, globalModel, maxDepthProcessed, confidence, tick, timeDelta, fillIn, textures);
        ferns.addFrame(&fillIn.imageTexture, &fillIn.vertexTexture, &fillIn.normalTexture, currPose, tick, fernThresh);

        gui.render(globalModel.model(), Vertex::SIZE);
        tick++;

    }

    if (sply)
    {
        std::cout<<"saving ply..";
        savePly(globalModel, saveply_file, confidence);
    }

    return 0;
}