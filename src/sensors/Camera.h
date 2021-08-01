#ifndef CAM_MODEL_TYPE
#define CAM_MODEL_TYPE
struct CameraModel
{
    float fx, fy, cx, cy, width, height;
    CameraModel()
     : fx(0), fy(0), cx(0), cy(0), width(0), height(0)
    {}

    CameraModel(float fx_, float fy_, float cx_, float cy_, float width_, float height_)
     : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(width_), height(height_)
    {}

    CameraModel operator()(int level) const
    {
        int div = 1 << level;
        return (CameraModel (fx / div, fy / div, cx / div, cy / div, width, height));
    }
};
#endif //CAM_MODEL_TYPE