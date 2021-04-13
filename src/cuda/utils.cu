#include "utils.cuh"
#include "convenience.cuh"
#include "operators.cuh"


__device__ float3 getVertex(int x, int y, int cx, int cy, float fx, float fy, float z)
{
    float3 vert  = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    vert.x = (x - cx) * z * fx;
    vert.y = (y - cy) * z * fy;
    vert.z = z;

    return vert;
}

__device__ float3 getNormal(float3 vPosition, int x, int y, int cx, int cy, int fx, int fy, float z)
{
    float3 vPosition_x  = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    float3 vPosition_y  = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    float3 del_x  = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    float3 del_y  = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    vPosition_x = getVertex(x + 1, y, cx, cy, fx, fy, z);
    vPosition_y = getVertex(x, y + 1, cx, cy, fx, fy, z);
    del_x = vPosition_x - vPosition;
    del_y = vPosition_y - vPosition;
    return normalized(cross(del_x, del_y));
}

__global__ void computeBilateralFilterKernel(const PtrStepSz<float> depth, PtrStepSz<float> filtered, const float depthCutoff)
{

    float sum1 = 0;
    float sum2 = 0;

    float out;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x > depth.cols && y >depth.rows)
        return;

    float value = depth.ptr(y)[x];


    if(value > depthCutoff || value < 0.3)
    {
        out = 0;
    }

    else
    {

        const float sigma_space2_inv_half = 0.024691358; // 0.5 / (sigma_space * sigma_space)
        const float sigma_color2_inv_half = 555.556; // 0.5 / (sigma_color * sigma_color)
                
        const int R = 6;
        const int D = R * 2 + 1;

        int tx = min(x - D / 2 + D, int(depth.cols));
        int ty = min(y - D / 2 + D, int(depth.rows));


        for(int cy = max(y - D / 2, 0); cy < ty; ++cy)
        {
            for(int cx = max(x - D / 2, 0); cx < tx; ++cx)
            {
                float tmp = depth.ptr(cy)[cx];
                
                
                float space2 = (float(x) - float(cx)) * (float(x) - float(cx)) + (float(y) - float(cy)) * (float(y) - float(cy));
                float color2 = (float(value) - float(tmp)) * (float(value) - float(tmp));

                float weight = exp(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

                sum1 += float(tmp) * weight;
                sum2 += weight;
            }
        }

    }
    out = sum1/sum2;
    filtered.ptr(y)[x] = out;

}


void computeBilateralFilter(const DeviceArray2D<float>& depth, DeviceArray2D<float> & filtered, const float depthCutoff)
{
    filtered.create (depth.rows (), depth.cols ());
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (depth.cols (), block.x);
    grid.y = getGridDim (depth.rows (), block.y);
    computeBilateralFilterKernel<<<grid, block>>>(depth, filtered, depthCutoff);
    cudaSafeCall (cudaGetLastError ());


}

__global__ void fillinImage(int width, int height, float* existingRgb, float* rawRgb, bool passthrough, float* dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        float4 sample;
        sample.x = existingRgb[y * width * 4 + (x * 4) + 0];
        sample.y = existingRgb[y * width * 4 + (x * 4) + 1];
        sample.z = existingRgb[y * width * 4 + (x * 4) + 2];
        sample.w = existingRgb[y * width * 4 + (x * 4) + 3];

        if (((sample.x + sample.y + sample.z) == 0) || passthrough == 1)
        {
            dst[y * width * 4 + (x * 4) + 0] = rawRgb[y * width * 3 + (x * 3) + 0];
            dst[y * width * 4 + (x * 4) + 1] = rawRgb[y * width * 3 + (x * 3) + 1];
            dst[y * width * 4 + (x * 4) + 2] = rawRgb[y * width * 3 + (x * 3) + 2];
            dst[y * width * 4 + (x * 4) + 3] = /*rawRgb[y * width * 4 + (x * 4) + 3]*/1;
        }
        else
        {            
            dst[y * width * 4 + (x * 4) + 0] = sample.x;
            dst[y * width * 4 + (x * 4) + 1] = sample.y;
            dst[y * width * 4 + (x * 4) + 2] = sample.z;
            dst[y * width * 4 + (x * 4) + 3] = sample.w;
        }
    }


}

void fillinRgb(int width, int height, float* existingRgb, float* rawRgb, bool passthrough, float* dst)
{

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    
    grid.x = getGridDim (width, block.x);
    grid.y = getGridDim (height, block.y);

    fillinImage<<<grid, block>>>(width, height, existingRgb, rawRgb, passthrough, dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void fillinVert(int width, int height, float cx, float cy, float fx_inv, float fy_inv,
                            PtrStepSz<float> existingVertex, PtrStepSz<float> rawDepth,bool passthrough, PtrStepSz<float> dst)
{
    // float halfPixX = 0.5 * (1.0 / width);
    // float halfPixY = 0.5 * (1.0 / height);

    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if((u > 0) && (u < width) && (v > 0) && (v < height))
    {
        float4 sample  = make_float4(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff),  __int_as_float(0x7fffffff));
        sample.x = existingVertex.ptr(v)[u];
        sample.y = existingVertex.ptr(v + height)[u];
        sample.z = existingVertex.ptr(v + 2 * height)[u];
        sample.w = existingVertex.ptr(v + 3 * height)[u];
         if((sample.z == 0) || (passthrough == 1))
         {
            float z = rawDepth.ptr(v)[u];
            if ((z < 0) || (z > 5) || isnan(z))
            {
                z = 0;
            }
            float3 vert  = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
            vert = getVertex(u, v, cx, cy, fx_inv, fy_inv, z);
            // printf("fx_inv, fy_inv = %f, %f\n", fx_inv, fy_inv);
            dst.ptr(v)[u] = vert.x; // (u - cx)*z*fx
            dst.ptr(v +  height)[u] = vert.y; // (v - cy)*z*fy
            dst.ptr(v + 2 * height)[u] = vert.z; // z
            dst.ptr(v + 3 * height)[u] = 1;
            // printf("u = %d v = %d\n", u, v);

         }
         else
         {
            dst.ptr(v)[u] = sample.x;
            dst.ptr(v +  height)[u] = sample.y;
            dst.ptr(v + 2 * height)[u] = sample.z;
            dst.ptr(v + 3 * height)[u] = sample.w;
            // printf("x = %f y = %f\n", sample.x, sample.y);
         }

    }
}



void fillinVertex(const CameraModel& intr, int width, int height, DeviceArray2D<float>& existingVertex,
                    DeviceArray2D<float>& rawDepth, bool passthrough, DeviceArray2D<float>& dst)
{
    dim3 block(32, 8);
    dim3 grid(1, 1, 1);

    grid.x = getGridDim (width, block.x);
    grid.y = getGridDim (height, block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    fillinVert<<<grid, block>>>(width, height, cx, cy, 1/fx, 1/fy, existingVertex, rawDepth, passthrough, dst);
    cudaSafeCall(cudaGetLastError());    
}

__global__ void fillinNorm(int width, int height, float cx, float cy, float fx, float fy,
                            PtrStepSz<float> existingNormal, PtrStepSz<float> rawDepth,bool passthrough, PtrStepSz<float> dst)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < width && v < height)
    {
        float4 sample  = make_float4 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff),  __int_as_float(0x7fffffff));
        sample.x = existingNormal.ptr(v)[u];
        sample.y = existingNormal.ptr(v+height)[u];
        sample.z = existingNormal.ptr(v + 2 * height)[u];
        sample.w = existingNormal.ptr(v + 3 * height)[u];
         if(sample.z == 0 || passthrough == 1)
         {
            float3 norm = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
            float3 vpos  = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
            vpos = getVertex(u, v, cx, cy, fx, fy, rawDepth.ptr(v)[u]);
            norm = getNormal(vpos, u, v, cx, cy, fx, fy, rawDepth.ptr(v)[u]);
            dst.ptr(v)[u] = norm.x;
            dst.ptr(v +  height)[u] = norm.y;
            dst.ptr(v + 2 * height)[u] = norm.z;
            dst.ptr(v + 3 * height)[u] = 1;

         }
         else
         {
            dst.ptr(v)[u] = sample.x;
            dst.ptr(v +  height)[u] = sample.y;
            dst.ptr(v + 2 * height)[u] = sample.z;
            dst.ptr(v + 3 * height)[u] = sample.w;
         }

    }    
}

void fillinNormal(const CameraModel& intr, int width, int height, DeviceArray2D<float>& existingNormal,
                    DeviceArray2D<float>& rawDepth, bool passthrough, DeviceArray2D<float>& dst)
{
    dim3 block(32, 8);
    dim3 grid(1, 1, 1);

    grid.x = getGridDim (width, block.x);
    grid.y = getGridDim (height, block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    fillinNorm<<<grid, block>>>(width, height, cx, cy, 1/fx, 1/fy, existingNormal, rawDepth, passthrough, dst);
    cudaSafeCall(cudaGetLastError());    
}