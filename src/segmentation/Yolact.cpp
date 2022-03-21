#include "Yolact.h"

Yolact::Yolact(/*int width, int height*/)
{
    // ncnn::Net yolact;
    yolact.opt.use_vulkan_compute = true;
    yolact.load_param("/home/developer/deps/ncnn/examples/yolact.param");
    yolact.load_model("/home/developer/deps/ncnn/examples/yolact.bin");
}

// Yolact::~Yolact()
// {

// }
inline float Yolact::intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void Yolact::qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void Yolact::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void Yolact::nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int Yolact::detect_yolact(std::vector<Object>& objects, int imgShareableHandle)
{
    
	// ncnn::Net yolact;
 //    yolact.opt.use_vulkan_compute = true;
 //    yolact.load_param("/home/developer/works/data/yolact.param");
 //    yolact.load_model("/home/developer/works/data/yolact.bin");

    // const int target_size = 550;

    int img_w = 550;
    int img_h = 550;

    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);

    // const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    // const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    // in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolact.create_extractor();
    ex.input("input.1", imgShareableHandle);
    ncnn::Mat maskmaps;
    ncnn::Mat location;
    ncnn::Mat mask;
    ncnn::Mat confidence;

    ex.extract_share("619", maskmaps, true); // 138x138 x 32
    ex.extract_share("816", location, true);   // 4 x 19248
    ex.extract_share("818", mask, true);       // maskdim 32 x 19248
    ex.extract_share("820", confidence, true); // 81 x 19248

    int num_class = confidence.w;
    int num_priors = confidence.h;

    // make priorbox
    ncnn::Mat priorbox(4, num_priors);
    {
        const int conv_ws[5] = {69, 35, 18, 9, 5};
        const int conv_hs[5] = {69, 35, 18, 9, 5};

        const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
        const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

        float* pb = priorbox;

        for (int p = 0; p < 5; p++)
        {
            int conv_w = conv_ws[p];
            int conv_h = conv_hs[p];

            float scale = scales[p];

            for (int i = 0; i < conv_h; i++)
            {
                for (int j = 0; j < conv_w; j++)
                {
                    // +0.5 because priors are in center-size notation
                    float cx = (j + 0.5f) / conv_w;
                    float cy = (i + 0.5f) / conv_h;

                    for (int k = 0; k < 3; k++)
                    {
                        float ar = aspect_ratios[k];

                        ar = sqrt(ar);

                        float w = scale * ar / 550;
                        float h = scale / ar / 550;

                        // This is for backward compatibility with a bug where I made everything square by accident
                        // cfg.backbone.use_square_anchors:
                        h = w;

                        pb[0] = cx;
                        pb[1] = cy;
                        pb[2] = w;
                        pb[3] = h;

                        pb += 4;
                    }
                }
            }
        }
    }

    const float confidence_thresh = 0.05f;
    const float nms_threshold = 0.5f;
    const int keep_top_k = 200;

    std::vector<std::vector<Object> > class_candidates;
    class_candidates.resize(num_class);

    for (int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence.row(i);
        const float* loc = location.row(i);
        const float* pb = priorbox.row(i);
        const float* maskdata = mask.row(i);

        // find class id with highest score
        // start from 1 to skip background
        int label = 0;
        float score = 0.f;
        for (int j = 1; j < num_class; j++)
        {
            float class_score = conf[j];
            if (class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        // ignore background or low score
        if (label == 0 || score <= confidence_thresh)
            continue;

        // CENTER_SIZE
        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float pb_cx = pb[0];
        float pb_cy = pb[1];
        float pb_w = pb[2];
        float pb_h = pb[3];

        float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
        float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
        float bbox_w = (float)(exp(var[2] * loc[2]) * pb_w);
        float bbox_h = (float)(exp(var[3] * loc[3]) * pb_h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        // clip
        obj_x1 = std::max(std::min(obj_x1 * 550, (float)(550 - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * 550, (float)(550 - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * 550, (float)(550 - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * 550, (float)(550 - 1)), 0.f);

        // append object
        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
        obj.label = label;
        obj.prob = score;
        obj.maskdata = std::vector<float>(maskdata, maskdata + mask.w);

        class_candidates[label].push_back(obj);
    }

    objects.clear();
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_bboxes(candidates, picked, nms_threshold);

        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            // objects.push_back(candidates[z]);
            if (candidates[z].prob>= .6)
            {
                objects.push_back(candidates[z]);
            }
        }
    }

    qsort_descent_inplace(objects);

    // keep_top_k
    if (keep_top_k < (int)objects.size())
    {
        objects.resize(keep_top_k);
    }

    // generate mask
    for (int i = 0; i < (int)objects.size(); i++)
    {
        Object& obj = objects[i];

        cv::Mat mask(maskmaps.h, maskmaps.w, CV_32FC1);
        {
            mask = cv::Scalar(0.f);

            for (int p = 0; p < maskmaps.c; p++)
            {
                const float* maskmap = maskmaps.channel(p);
                float coeff = obj.maskdata[p];
                float* mp = (float*)mask.data;

                // mask += m * coeff
                for (int j = 0; j < maskmaps.w * maskmaps.h; j++)
                {
                    mp[j] += maskmap[j] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y = 0; y < img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x = 0; x < img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    // cout << "mask size " <<img_w <<" height " << img_h<< endl;

    return 0;
}


cv::Mat Yolact::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::vector<Object>& track_objects)
{
    static const char* class_names[] = {"background",
                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"
                                       };

    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}
    };

    cv::Mat image = bgr.clone();

    int color_index = 1;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.6)
            continue;
        track_objects.push_back(objects[i]);
        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        // const unsigned char* color = colors[color_index % 81];
        // color_index++;

        // cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        // sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        //fprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        // cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
        //               cv::Scalar(255, 255, 255), -1);

        // cv::putText(image, text, cv::Point(x, y + label_size.height),
        //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // draw mask
        for (int y = 0; y < image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    // p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    // p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    // p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                    p[x] = cv::saturate_cast<unsigned char>(color_index);
                }
                // p += 3;
            }
        }
        color_index++;
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640,480), cv::INTER_LINEAR);
    // cv::imshow(" instance ", image);
    // cv::waitKey(0);
    // return 0;
    return resized_image;
}

volatile bool done = false;
cv::Mat Yolact::processFrame(int fd){

    std::vector<Object> objects;
    std::vector<Object> track_objects;
    detect_yolact(objects, fd);
    cv::Mat mask = draw_objects(Mat::zeros(Size(550,550),CV_8UC1), objects, track_objects);
    return mask;

}


void Yolact::computeBBox(std::vector<Object> objects, GLfloat *& bbox_verts_ptr, GLushort *& bbox_ele_ptr,  int* no, unsigned short* depth, float cx, float cy, float fx, float fy, float width, float height)
{

    int num_objects = objects.size();
    // int box_attrbs_num = 33; // 32(box cords)+1(color) 
    int box_attrbs_num = 32; 

    *no = num_objects;

    bbox_verts_ptr = new GLfloat[num_objects*box_attrbs_num];
    bbox_ele_ptr = new GLushort[num_objects*24];

    float obj_depth = 0.5;
    int d_index = 0;
    // (x - cam.x) * z * cam.z, (y - cam.y) * z * cam.w, z
    for (int i=0; i<num_objects;i++)
    {
        d_index = (int)(640 * (objects[i].rect.y + objects[i].rect.height/2)*480/550 + (objects[i].rect.x + objects[i].rect.width/2)*640/550);
        obj_depth = depth[d_index]/1000;
        bbox_verts_ptr[i*box_attrbs_num] = (((objects[i].rect.x * 640/550)- cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 1] =(((objects[i].rect.y * 480/550) - cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 2] = obj_depth;  
        bbox_verts_ptr[i*box_attrbs_num + 3] = 2;  


        bbox_verts_ptr[i*box_attrbs_num + 4] = ((((objects[i].rect.x+objects[i].rect.width) * 640/550) - cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 5] = (((objects[i].rect.y * 480/550) - cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 6] = obj_depth;  
        bbox_verts_ptr[i*box_attrbs_num + 7] = 2;  


        bbox_verts_ptr[i*box_attrbs_num + 8] = ((((objects[i].rect.x+objects[i].rect.width) * 640/550)- cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 9] = ((((objects[i].rect.y+objects[i].rect.height) * 480/550)- cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 10] = obj_depth;  
        bbox_verts_ptr[i*box_attrbs_num + 11] = 2;  


        bbox_verts_ptr[i*box_attrbs_num + 12] = (((objects[i].rect.x * 640/550)- cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 13] = ((((objects[i].rect.y+objects[i].rect.height) * 480/550)- cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 14] = obj_depth;  
        bbox_verts_ptr[i*box_attrbs_num + 15] = 2; 



        bbox_verts_ptr[i*box_attrbs_num + 16] = (((objects[i].rect.x * 640/550)- cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 17] =(((objects[i].rect.y * 480/550) - cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 18] = obj_depth + 0.5;  
        bbox_verts_ptr[i*box_attrbs_num + 19] = 2;  


        bbox_verts_ptr[i*box_attrbs_num + 20] = ((((objects[i].rect.x+objects[i].rect.width) * 640/550) - cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 21] = (((objects[i].rect.y * 480/550) - cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 22] = obj_depth + 0.5;  
        bbox_verts_ptr[i*box_attrbs_num + 23] = 2; 



        bbox_verts_ptr[i*box_attrbs_num + 24] = ((((objects[i].rect.x+objects[i].rect.width) * 640/550)- cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 25] = ((((objects[i].rect.y+objects[i].rect.height) * 480/550)- cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 26] = obj_depth + 0.5;  
        bbox_verts_ptr[i*box_attrbs_num + 27] = 2;  


        bbox_verts_ptr[i*box_attrbs_num + 28] = (((objects[i].rect.x * 640/550)- cx) * obj_depth * 1/fx);
        bbox_verts_ptr[i*box_attrbs_num + 29] = ((((objects[i].rect.y+objects[i].rect.height) * 480/550)- cy) * obj_depth * 1/fy);
        bbox_verts_ptr[i*box_attrbs_num + 30] = obj_depth + 0.5;  
        bbox_verts_ptr[i*box_attrbs_num + 31] = 2; 

        
        //box colour
        // bbox_verts_ptr[i*box_attrbs_num + 32] = 50; 

        
        bbox_ele_ptr[i*24] = i*8; 
        bbox_ele_ptr[i*24 + 1] = i*8 + 1; 
        bbox_ele_ptr[i*24 + 2] = i*8 + 1; 
        bbox_ele_ptr[i*24 + 3] = i*8 + 2; 

        bbox_ele_ptr[i*24 + 4] = i*8 + 2; 
        bbox_ele_ptr[i*24 + 5] = i*8 + 3; 
        bbox_ele_ptr[i*24 + 6] = i*8 + 3; 
        bbox_ele_ptr[i*24 + 7] = i*8 ; 

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

    };
}