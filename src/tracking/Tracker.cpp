#include "Tracker.h"

Tracker::Tracker()
{
    //TO DO : change this to a function
    idx_to_rgb = {
                    {0,Eigen::Vector3f(0,0,1)},
                    {1,Eigen::Vector3f(0,1,0)},
                    {2,Eigen::Vector3f(1,0,0)},
                    {3,Eigen::Vector3f(0,1,1)},
                    {4,Eigen::Vector3f(0.5,0.4,0.2)},
                    {5,Eigen::Vector3f(0,0,0.5)},
                    {6,Eigen::Vector3f(0,0.5,0)},
                    {7,Eigen::Vector3f(0.5,0,0)},
                    {8,Eigen::Vector3f(0.5,0,1)},
                    {9,Eigen::Vector3f(0,0.5,0.5)},
                    {10,Eigen::Vector3f(1,0.25,0.5)},
                    {11,Eigen::Vector3f(.66,0.25,0.5)},
                    {12,Eigen::Vector3f(1,0.25,0.5)},                  
                    {13,Eigen::Vector3f(1,0,0.65)},                  
                    {15,Eigen::Vector3f(1,0.45,0.85)},                  
                    {14,Eigen::Vector3f(1,0.25,0.25)},                  

                };
    b_idx = 0;
}

void Tracker::Update(vector<Point3f>& detections)
{

    if(tracks.size()==0)
    {
        for(int i=0;i<detections.size();i++)
        {
            kalman_track* tr=new kalman_track(NextID,detections[i],dt,acceleration);
    		NextID++;
            tracks.push_back(tr);
        }   
    }


    int N=tracks.size();        
    int M=detections.size();    


    vector< vector<double> > Cost(N,vector<double>(M));
    vector<int> assignment;

    double dist;
    for(int i=0;i<tracks.size();i++)
    {   

        for(int j=0;j<detections.size();j++)
        {
            Point3d diff=(tracks[i]->prediction-detections[j]);
            dist=sqrtf(diff.x*diff.x+diff.y*diff.y + diff.z*diff.z);
            Cost[i][j]=dist;
        }
    }
    // -----------------------------------
    // Solving assignment problem (tracks and predictions of Kalman filter)
    // -----------------------------------
    AssignmentProblemSolver APS;
    APS.Solve(Cost,assignment,AssignmentProblemSolver::optimal);
    vector<int> not_assigned_tracks;

    for(int i=0;i<assignment.size();i++)
    {
        if(assignment[i]!=-1)
        {
            if(Cost[i][assignment[i]]>max_distance)
            {
                assignment[i]=-1;
                not_assigned_tracks.push_back(i);
            }
        }
        else
        {           
            tracks[i]->misses++;
        }

    }

    // -----------------------------------
    // If track didn't get detects long time, remove it.
    // -----------------------------------
    for(int i=0;i<tracks.size();i++)
    {
        if(tracks[i]->misses>max_misses)
        {
            delete tracks[i];
            tracks.erase(tracks.begin()+i);
            assignment.erase(assignment.begin()+i);
            i--;
        }
    }
    // -----------------------------------
    // Search for unassigned detects
    // -----------------------------------
    vector<int> not_assigned_detections;
    vector<int>::iterator it;
    for(int i=0;i<detections.size();i++)
    {
        it=find(assignment.begin(), assignment.end(), i);
        if(it==assignment.end())
        {
            not_assigned_detections.push_back(i);
        }
    }

    // -----------------------------------
    // and start new tracks for them.
    // -----------------------------------
    if(not_assigned_detections.size()!=0)
    {
        for(int i=0;i<not_assigned_detections.size();i++)
        {
            kalman_track* tr=new kalman_track(NextID, detections[not_assigned_detections[i]],dt,acceleration);
    		NextID++;
            tracks.push_back(tr);
        }   
    }

    // Update Kalman Filters state

    for(int i=0;i<assignment.size();i++)
    {

        tracks[i]->KF->GetPrediction();

        if(assignment[i]!=-1)
        {
            tracks[i]->misses=0;
            tracks[i]->prediction=tracks[i]->KF->Update(detections[assignment[i]],1);
        }else               
        {
            tracks[i]->prediction=tracks[i]->KF->Update(Point3f(0,0,0),0);    
        }
        
        if(tracks[i]->trace.size()>max_trace)
        {
            tracks[i]->trace.erase(tracks[i]->trace.begin(),tracks[i]->trace.end()-max_trace);
        }

        tracks[i]->trace.push_back(tracks[i]->prediction);
        tracks[i]->KF->LastResult=tracks[i]->prediction;
    }

}

float Tracker::distance(int x1, int y1, int z1, int x2, int y2, int z2)
{
    // Calculating distance
    return sqrt(pow(x2 - x1, 2) +
                pow(y2 - y1, 2) +
                pow(z2 - z1, 2));

}

float Tracker::encodeColor(Eigen::Vector3f c)
{
    int rgb = int(round(c(0) * 255));
    rgb = (rgb << 8) + int(round(c(1) * 255));
    rgb = (rgb << 8) + int(round(c(2) * 255));
    return float(rgb);
}

Eigen::Vector3f Tracker::decodeColor(float c)
{
    Eigen::Vector3f col;
    col(0) = float(int(c) >> 16 & 0xFF) / 255;
    col(1) = float(int(c) >> 8 & 0xFF) / 255;
    col(2) = float(int(c) & 0xFF) / 255;
    return col;
}

void Tracker::getBoxVBO(int& ts, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr)
{   
    ts = tracks.size();
    if (ts <=0 )
        return;
    bbox_verts_ptr = new GLfloat[ts*8*4];
    bbox_ele_ptr = new GLushort[ts*24];

    for (int i = 0; i < ts; ++i)
    {
        bbox_verts_ptr[i*8*4] = tracks[i]->bbox_vertices_ptr[0];
        bbox_verts_ptr[i*8*4 + 1]= tracks[i]->bbox_vertices_ptr[1]; 
        bbox_verts_ptr[i*8*4 + 2]= tracks[i]->bbox_vertices_ptr[2]; 
        bbox_verts_ptr[i*8*4 + 3]= tracks[i]->bbox_vertices_ptr[3]; 

        bbox_verts_ptr[i*8*4 + 4]= tracks[i]->bbox_vertices_ptr[4]; 
        bbox_verts_ptr[i*8*4 + 5]= tracks[i]->bbox_vertices_ptr[5]; 
        bbox_verts_ptr[i*8*4 + 6]= tracks[i]->bbox_vertices_ptr[6]; 
        bbox_verts_ptr[i*8*4 + 7]= tracks[i]->bbox_vertices_ptr[7];

        bbox_verts_ptr[i*8*4 + 8]= tracks[i]->bbox_vertices_ptr[8]; 
        bbox_verts_ptr[i*8*4 + 9]= tracks[i]->bbox_vertices_ptr[9];
        bbox_verts_ptr[i*8*4 + 10]= tracks[i]->bbox_vertices_ptr[10]; 
        bbox_verts_ptr[i*8*4 + 11]= tracks[i]->bbox_vertices_ptr[11];

        bbox_verts_ptr[i*8*4 + 12]= tracks[i]->bbox_vertices_ptr[12]; 
        bbox_verts_ptr[i*8*4 + 13]= tracks[i]->bbox_vertices_ptr[13]; 
        bbox_verts_ptr[i*8*4 + 14]= tracks[i]->bbox_vertices_ptr[14];
        bbox_verts_ptr[i*8*4 + 15]= tracks[i]->bbox_vertices_ptr[15]; 

        bbox_verts_ptr[i*8*4 + 16]= tracks[i]->bbox_vertices_ptr[16]; 
        bbox_verts_ptr[i*8*4 + 17]= tracks[i]->bbox_vertices_ptr[17]; 
        bbox_verts_ptr[i*8*4 + 18]= tracks[i]->bbox_vertices_ptr[18]; 
        bbox_verts_ptr[i*8*4 + 19]= tracks[i]->bbox_vertices_ptr[19]; 

        bbox_verts_ptr[i*8*4 + 20]= tracks[i]->bbox_vertices_ptr[20]; 
        bbox_verts_ptr[i*8*4 + 21]= tracks[i]->bbox_vertices_ptr[21]; 
        bbox_verts_ptr[i*8*4 + 22]= tracks[i]->bbox_vertices_ptr[22]; 
        bbox_verts_ptr[i*8*4 + 23]= tracks[i]->bbox_vertices_ptr[23]; 

        bbox_verts_ptr[i*8*4 + 24]= tracks[i]->bbox_vertices_ptr[24]; 
        bbox_verts_ptr[i*8*4 + 25]= tracks[i]->bbox_vertices_ptr[25]; 
        bbox_verts_ptr[i*8*4 + 26]= tracks[i]->bbox_vertices_ptr[26]; 
        bbox_verts_ptr[i*8*4 + 27]= tracks[i]->bbox_vertices_ptr[27];

        bbox_verts_ptr[i*8*4 + 28]= tracks[i]->bbox_vertices_ptr[28]; 
        bbox_verts_ptr[i*8*4 + 29]= tracks[i]->bbox_vertices_ptr[29];
        bbox_verts_ptr[i*8*4 + 30]= tracks[i]->bbox_vertices_ptr[30]; 
        bbox_verts_ptr[i*8*4 + 31]= tracks[i]->bbox_vertices_ptr[31]; 


        bbox_ele_ptr[i*24] = i*8; 
        bbox_ele_ptr[i*24 + 1] = i*8 + 1; 
        bbox_ele_ptr[i*24 + 2] = i*8 + 1; 
        bbox_ele_ptr[i*24 + 3] = i*8 + 2; 

        bbox_ele_ptr[i*24 + 4] = i*8 + 2; 
        bbox_ele_ptr[i*24 + 5] = i*8 + 3; 
        bbox_ele_ptr[i*24 + 6] = i*8 + 3; 
        bbox_ele_ptr[i*24 + 7] = i*8; 

        bbox_ele_ptr[i*24 + 8] = i*8 + 4; 
        bbox_ele_ptr[i*24 + 9] = i*8 + 5; 
        bbox_ele_ptr[i*24 + 10] = i*8 + 5; 
        bbox_ele_ptr[i*24 + 11] = i*8 + 6; 

        bbox_ele_ptr[i*24 + 12] = i*8 + 6; 
        bbox_ele_ptr[i*24 + 13] = i*8 + 7; 
        bbox_ele_ptr[i*24 + 14] = i*8 + 7; 
        bbox_ele_ptr[i*24 + 15] = i*8 + 4; 

        bbox_ele_ptr[i*24 + 16] = i*8; 
        bbox_ele_ptr[i*24 + 17] = i*8 + 4; 
        bbox_ele_ptr[i*24 + 18] = i*8 + 3; 
        bbox_ele_ptr[i*24 + 19] = i*8 + 7; 

        bbox_ele_ptr[i*24 + 20] = i*8 + 1; 
        bbox_ele_ptr[i*24 + 21] = i*8 + 5; 
        bbox_ele_ptr[i*24 + 22] = i*8 + 2; 
        bbox_ele_ptr[i*24 + 23] = i*8 + 6; 
    }

}


void Tracker::Update(std::vector<Object> objects, const Eigen::Matrix4f & currPose, int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height)
{
    obj_tid = new int[objects.size()];

    track_objects = objects;
    if (track_objects.size()>0)
    {
        vector<Point3f> centers;

        float obj_depth = 0.5;
        int d_index = 0;

        for (size_t i = 0; i < track_objects.size(); i++)
        {
            const Object& track_obj = track_objects[i];
            if (track_obj.prob > 0.65)
            {

                d_index = (int)(640 * (track_obj.rect.y + track_obj.rect.height/2)*480/550 + (track_obj.rect.x + track_obj.rect.width/2)*640/550);
                obj_depth = depth[d_index]/1000;
                if (isnan(obj_depth))
                    obj_depth = 0;
                Point3f center = Point3f(track_obj.rect.x+(track_obj.rect.width/2), track_obj.rect.y+(track_obj.rect.height/2), obj_depth);
                centers.push_back(center);
            }
            obj_tid[i] = -1; // initialize obj_id map to -1
        }

        if (centers.size()>0)
        {
            int num_objects = centers.size();

            Update(centers);
            for(int i=0;i<tracks.size();i++)
            {
                if(tracks[i]->trace.size()>1)
                {
                    for(int j=0;j<tracks[i]->trace.size()-1;j++)
                    {
                        for(int ik=0;ik<centers.size();ik++)
                        {
                            if(distance(tracks[i]->prediction.x,tracks[i]->prediction.y, tracks[i]->prediction.z, centers[ik].x, centers[ik].y, centers[ik].z)< 10)
                            {
                                const Object& track_obj = track_objects[ik];
                                obj_tid[ik] = tracks[i]->track_id;
                                //TO DO
                                Eigen::Vector3f rgb;
                                if (tracks[i]->track_id > 13)
                                {
                                    rgb(0)=1;
                                    rgb(1)=1;
                                    rgb(2)=1;
                                }
                                else
                                {
                                    rgb = idx_to_rgb[tracks[i]->track_id];
                                }
                                float ec = encodeColor(rgb);

                                d_index = (int)(640 * (track_obj.rect.y + track_obj.rect.height/2)*480/550 + (track_obj.rect.x + track_obj.rect.width/2)*640/550);
                                obj_depth = depth[d_index]/1000;
                                if (isnan(obj_depth))
                                    obj_depth = 0;

                                Eigen::Vector4f p = currPose * Eigen::Vector4f((((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx),
                                                                               (((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy),
                                                                                obj_depth,
                                                                                1);
                                tracks[i]->bbox_vertices_ptr[0] = p[0];
                                tracks[i]->bbox_vertices_ptr[1] = p[1];
                                tracks[i]->bbox_vertices_ptr[2] = p[2];  
                                tracks[i]->bbox_vertices_ptr[3] = ec;  

                                p = currPose * Eigen::Vector4f(((((track_obj.rect.x+track_obj.rect.width) * 640/550) - cx) * obj_depth * 1/fx),
                                                                               (((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy),
                                                                                obj_depth,
                                                                                1);

                                tracks[i]->bbox_vertices_ptr[4] = p[0];
                                tracks[i]->bbox_vertices_ptr[5] = p[1];
                                tracks[i]->bbox_vertices_ptr[6] = p[2];  
                                tracks[i]->bbox_vertices_ptr[7] = ec;  


                                p = currPose * Eigen::Vector4f(((((track_obj.rect.x+track_obj.rect.width) * 640/550)- cx) * obj_depth * 1/fx),
                                                                               ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy),
                                                                                obj_depth,
                                                                                1);

                                tracks[i]->bbox_vertices_ptr[8] = p[0];
                                tracks[i]->bbox_vertices_ptr[9] = p[1];
                                tracks[i]->bbox_vertices_ptr[10] = p[2];  
                                tracks[i]->bbox_vertices_ptr[11] = ec;  


                                p = currPose * Eigen::Vector4f((((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx),
                                                                               ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy),
                                                                                obj_depth,
                                                                                1);

                                tracks[i]->bbox_vertices_ptr[12] = p[0];
                                tracks[i]->bbox_vertices_ptr[13] = p[1];
                                tracks[i]->bbox_vertices_ptr[14] = p[2];  
                                tracks[i]->bbox_vertices_ptr[15] = ec; 


                                p = currPose * Eigen::Vector4f((((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx),
                                                                            (((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy),
                                                                                obj_depth - 0.5,
                                                                                1);

                                tracks[i]->bbox_vertices_ptr[16] = p[0]; 
                                tracks[i]->bbox_vertices_ptr[17] = p[1];
                                tracks[i]->bbox_vertices_ptr[18] = p[2];    
                                tracks[i]->bbox_vertices_ptr[19] = ec;  

                                p = currPose * Eigen::Vector4f(((((track_obj.rect.x+track_obj.rect.width) * 640/550) - cx) * obj_depth * 1/fx),
                                                                               (((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy),
                                                                                obj_depth - 0.5,
                                                                                1);


                                tracks[i]->bbox_vertices_ptr[20] = p[0]; 
                                tracks[i]->bbox_vertices_ptr[21] = p[1]; 
                                tracks[i]->bbox_vertices_ptr[22] = p[2];   
                                tracks[i]->bbox_vertices_ptr[23] = ec; 

                                p = currPose * Eigen::Vector4f(((((track_obj.rect.x+track_obj.rect.width) * 640/550)- cx) * obj_depth * 1/fx),
                                                                              ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy),
                                                                                obj_depth - 0.5,
                                                                                1);


                                tracks[i]->bbox_vertices_ptr[24] = p[0];
                                tracks[i]->bbox_vertices_ptr[25] = p[1];
                                tracks[i]->bbox_vertices_ptr[26] = p[2];   
                                tracks[i]->bbox_vertices_ptr[27] = ec;  

                                p = currPose * Eigen::Vector4f((((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx),
                                                                               ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy),
                                                                                obj_depth - 0.5,
                                                                                1);

                                tracks[i]->bbox_vertices_ptr[28] =p[0]; 
                                tracks[i]->bbox_vertices_ptr[29] =p[1];
                                tracks[i]->bbox_vertices_ptr[30] =p[2];  
                                tracks[i]->bbox_vertices_ptr[31] = ec; 

                                // b_idx++;

                            }
                        }
                    }
                    for(int x=0;x<tracks.size();x++)
                    {
                        start_time = tracks[x]->begin_time;
                        if((clock() - start_time) > 10000 )                              
                        {
                            if(tracks[x]->suspicious != 1)
                            {
                                tracks[x]->suspicious = 1;
                            }                                          
                        }
                    }
                }
            }
        }
        // *no = b_idx;
    }

}

kalman_track::kalman_track(int td, Point3f pt, float dt, float acceleration)
{

    track_id=td;
    
    KF = new TKalmanFilter(pt,dt,acceleration);
    begin_time = clock();
    suspicious = 0;
    prediction=pt;
    misses=0;
    for(int i = 0; i< 8*4; i++)
    {
        bbox_vertices_ptr[i] = 0;
        if (i < 24)
            bbox_elements_ptr[i] = 0;
    }
    ec = 0;

}

kalman_track::~kalman_track()
{
    delete KF;
}
