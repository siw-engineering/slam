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

}

void Tracker::init_track_objects(std::vector<Object> objs)
{
    track_objects = objs;
}

void Tracker::Update(vector<Point2f>& detections)
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
            Point2d diff=(tracks[i]->prediction-detections[j]);
            dist=sqrtf(diff.x*diff.x+diff.y*diff.y);
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
            tracks[i]->prediction=tracks[i]->KF->Update(Point2f(0,0),0);    
        }
        
        if(tracks[i]->trace.size()>max_trace)
        {
            tracks[i]->trace.erase(tracks[i]->trace.begin(),tracks[i]->trace.end()-max_trace);
        }

        tracks[i]->trace.push_back(tracks[i]->prediction);
        tracks[i]->KF->LastResult=tracks[i]->prediction;
    }

}

float Tracker::distance(int x1, int y1, int x2, int y2)
{
    // Calculating distance
    return sqrt(pow(x2 - x1, 2) +
                pow(y2 - y1, 2) * 1.0);

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


void Tracker::Update(std::vector<Object> objects, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr,  int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height)
{
    if (track_objects.size()>0)
    {
        vector<Point2f> centers;
        for (size_t i = 0; i < track_objects.size(); i++)
        {
            const Object& track_obj = track_objects[i];
            if (track_obj.prob > 0.65)
            {
                Point center = Point(track_obj.rect.x+(track_obj.rect.width/2), track_obj.rect.y+(track_obj.rect.height/2));
                centers.push_back(center);
            }
        }

        float obj_depth = 0.5;
        int d_index = 0;
        int b_idx = 0;

        if (centers.size()>0)
        {
            int num_objects = centers.size();
            int box_attrbs_num = 32; 
            bbox_verts_ptr = new GLfloat[(num_objects+1)*box_attrbs_num];
            bbox_ele_ptr = new GLushort[(num_objects+1)*24];
            Update(centers);
            for(int i=0;i<tracks.size();i++)
            {
                if(tracks[i]->trace.size()>1)
                {
                    for(int j=0;j<tracks[i]->trace.size()-1;j++)
                    {
                        for(int ik=0;ik<centers.size();ik++)
                        {
                            if(distance(tracks[i]->prediction.x,tracks[i]->prediction.y, centers[ik].x, centers[ik].y)< 10)
                            {
                                const Object& track_obj = track_objects[ik];
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
                                bbox_verts_ptr[b_idx*box_attrbs_num] = (((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 1] =(((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 2] = obj_depth;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 3] = ec;  


                                bbox_verts_ptr[b_idx*box_attrbs_num + 4] = ((((track_obj.rect.x+track_obj.rect.width) * 640/550) - cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 5] = (((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 6] = obj_depth;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 7] = ec;  


                                bbox_verts_ptr[b_idx*box_attrbs_num + 8] = ((((track_obj.rect.x+track_obj.rect.width) * 640/550)- cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 9] = ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 10] = obj_depth;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 11] = ec;  


                                bbox_verts_ptr[b_idx*box_attrbs_num + 12] = (((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 13] = ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 14] = obj_depth;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 15] = ec; 



                                bbox_verts_ptr[b_idx*box_attrbs_num + 16] = (((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 17] =(((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 18] = obj_depth - 0.5;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 19] = ec;  


                                bbox_verts_ptr[b_idx*box_attrbs_num + 20] = ((((track_obj.rect.x+track_obj.rect.width) * 640/550) - cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 21] = (((track_obj.rect.y * 480/550) - cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 22] = obj_depth - 0.5;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 23] = ec; 



                                bbox_verts_ptr[b_idx*box_attrbs_num + 24] = ((((track_obj.rect.x+track_obj.rect.width) * 640/550)- cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 25] = ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 26] = obj_depth - 0.5;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 27] = ec;  


                                bbox_verts_ptr[b_idx*box_attrbs_num + 28] = (((track_obj.rect.x * 640/550)- cx) * obj_depth * 1/fx);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 29] = ((((track_obj.rect.y+track_obj.rect.height) * 480/550)- cy) * obj_depth * 1/fy);
                                bbox_verts_ptr[b_idx*box_attrbs_num + 30] = obj_depth - 0.5;  
                                bbox_verts_ptr[b_idx*box_attrbs_num + 31] = ec; 


                                bbox_ele_ptr[b_idx*24] = b_idx*8; 
                                bbox_ele_ptr[b_idx*24 + 1] = b_idx*8 + 1; 
                                bbox_ele_ptr[b_idx*24 + 2] = b_idx*8 + 1; 
                                bbox_ele_ptr[b_idx*24 + 3] = b_idx*8 + 2; 

                                bbox_ele_ptr[b_idx*24 + 4] = b_idx*8 + 2; 
                                bbox_ele_ptr[b_idx*24 + 5] = b_idx*8 + 3; 
                                bbox_ele_ptr[b_idx*24 + 6] = b_idx*8 + 3; 
                                bbox_ele_ptr[b_idx*24 + 7] = b_idx*8 ; 

                                bbox_ele_ptr[b_idx*24 + 8] = b_idx*8 + 4; 
                                bbox_ele_ptr[b_idx*24 + 9] = b_idx*8 + 5; 
                                bbox_ele_ptr[b_idx*24 + 10] = b_idx*8 + 5; 
                                bbox_ele_ptr[b_idx*24 + 11] = b_idx*8 + 6; 

                                bbox_ele_ptr[b_idx*24 + 12] = b_idx*8 + 6; 
                                bbox_ele_ptr[b_idx*24 + 13] = b_idx*8 + 7; 
                                bbox_ele_ptr[b_idx*24 + 14] = b_idx*8 + 7; 
                                bbox_ele_ptr[b_idx*24 + 15] = b_idx*8 + 4; 

                                bbox_ele_ptr[b_idx*24 + 16] = b_idx*8; 
                                bbox_ele_ptr[b_idx*24 + 17] = b_idx*8 + 4; 
                                bbox_ele_ptr[b_idx*24 + 18] = b_idx*8 + 3; 
                                bbox_ele_ptr[b_idx*24 + 19] = b_idx*8 + 7; 

                                bbox_ele_ptr[b_idx*24 + 20] = b_idx*8 + 1; 
                                bbox_ele_ptr[b_idx*24 + 21] = b_idx*8 + 5; 
                                bbox_ele_ptr[b_idx*24 + 22] = b_idx*8 + 2; 
                                bbox_ele_ptr[b_idx*24 + 23] = b_idx*8 + 6; 

                                b_idx++;

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
        *no = b_idx;
    }
}

kalman_track::kalman_track(int td, Point2f pt, float dt, float acceleration)
{

    track_id=td;
    
    KF = new TKalmanFilter(pt,dt,acceleration);
 
    begin_time = clock();
    suspicious = 0;
    prediction=pt;
    misses=0;

}

kalman_track::~kalman_track()
{
    delete KF;
}
