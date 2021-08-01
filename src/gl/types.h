#include <pangolin/pangolin.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#ifndef GL_TYPES
#define GL_TYPES

class GPUTexture
{
    public:
        GPUTexture(const int width,
                   const int height,
                   const GLenum internalFormat,
                   const GLenum format,
                   const GLenum dataType,
                   const bool draw = false,
                   const bool cuda = false);

        virtual ~GPUTexture();

        static const std::string RGB, DEPTH_RAW, DEPTH_FILTERED, DEPTH_METRIC, DEPTH_METRIC_FILTERED, DEPTH_NORM;

        pangolin::GlTexture * texture;

        cudaGraphicsResource * cudaRes;

        const bool draw;

    private:
        GPUTexture() : texture(0), cudaRes(0), draw(false), width(0), height(0), internalFormat(0), format(0), dataType(0) {}
        const int width;
        const int height;
        const GLenum internalFormat;
        const GLenum format;
        const GLenum dataType;
};

#endif //GL_TYPES