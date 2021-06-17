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

__device__ float3 getNormal(float3 vPosition, int x, int y, int cx, int cy, float fx, float fy, float z)
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
                            PtrStepSz<float> existingVertex, PtrStepSz<float> rawDepth, bool passthrough, PtrStepSz<float> dst)
{
    // float halfPixX = 0.5 * (1.0 / width);
    // float halfPixY = 0.5 * (1.0 / height);

    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if((u < width) && (v < height))
    {
        float4 sample  = make_float4(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff),  __int_as_float(0x7fffffff));
        sample.x = existingVertex.ptr(v)[u];
        sample.y = existingVertex.ptr(v + height)[u];
        sample.z = existingVertex.ptr(v + 2 * height)[u];
        sample.w = existingVertex.ptr(v + 3 * height)[u];

         if((sample.z == 0) || (passthrough == 1))
         {
            float z = rawDepth.ptr(v)[u];
            if ((z < 0) || (z > 5) || isnan(z) || isinf(z))
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

    float* vertices_fillin = new float[height*width*4];
    dst.upload(&vertices_fillin[0], sizeof(float)*width, 4*height, width);
    delete[] vertices_fillin;


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

    float* normal_fillin = new float[height*width*4];
    dst.upload(&normal_fillin[0], sizeof(float)*width, 4*height, width);
    delete[] normal_fillin;


    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    fillinNorm<<<grid, block>>>(width, height, cx, cy, 1/fx, 1/fy, existingNormal, rawDepth, passthrough, dst);
    cudaSafeCall(cudaGetLastError());    
}


template<bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows, PtrStepSz<float> input, PtrStep<float> output, const int factor)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols  || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * factor;
    int ys = y * factor;

    // if (xs > input.cols - 1 || ys + (2*srows)  > input.rows - 1 )
    //     return;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output.ptr (y)[x] = qnan;
        // output.ptr (y)[x] = (unsigned char)0;
        return;
    }
    else
    {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        if (normalize)
            n = normalized (n);
        // output.ptr(y)[x] = n;
        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
        output.ptr (y + 3 * drows)[x] = 0;
    }
}


template<bool normalize>
void ResizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output, const int factor)
{
    int in_cols = input.cols ();
    int in_rows = input.rows () / 4;

    int out_cols = output.cols ();
    int out_rows = output.rows () / 4;

    // output.create (out_rows * 3, out_cols);

    dim3 block (32, 8);
    dim3 grid (getGridDim (out_cols, block.x), getGridDim (out_rows, block.y));
    resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output, factor);
    cudaCheckError();
}

void ResizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output, const int factor)
{
    ResizeMap<false>(input, output, factor);
}

void ResizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output, const int factor)
{
    ResizeMap<true>(input, output, factor);
}

__global__ void resizeImg(const int height, const int width, unsigned char * dst, float * src, const int factor)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    // const float qnan = __int_as_float(0x7fffffff);

    int xs = x * factor;
    int ys = y * factor;

    float x00 = src[(ys + 0) * (width*factor) * 4 + ((xs + 0) * 4) + 0];
    float x01 = src[(ys + 0) * (width*factor) * 4 + ((xs + 1) * 4) + 0];
    float x10 = src[(ys + 1) * (width*factor) * 4 + ((xs + 0) * 4) + 0];
    float x11 = src[(ys + 1) * (width*factor) * 4 + ((xs + 1) * 4) + 0];

    // if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    // {
    //     dst[y * width * 4 + (x * 4) + 0] = 0;
    //     dst[y * width * 4 + (x * 4) + 1] = 0;
    //     dst[y * width * 4 + (x * 4) + 2] = 0;
    //     // dst[y * width * 4 + (x * 4) + 3] = 0;
    //     return;
    // }
    // else
    // {
        float3 out = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        out.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = src[(ys + 0) * (width*factor) * 4 + ((xs + 0) * 4) + 1];
        float y01 = src[(ys + 0) * (width*factor) * 4 + ((xs + 1) * 4) + 1];
        float y10 = src[(ys + 1) * (width*factor) * 4 + ((xs + 0) * 4) + 1];
        float y11 = src[(ys + 1) * (width*factor) * 4 + ((xs + 1) * 4) + 1];
        out.y = (y00 + y01 + y10 + y11) / 4;


        float z00 = src[(ys + 0) * (width*factor) * 4 + ((xs + 0) * 4) + 2];
        float z01 = src[(ys + 0) * (width*factor) * 4 + ((xs + 1) * 4) + 2];
        float z10 = src[(ys + 1) * (width*factor) * 4 + ((xs + 0) * 4) + 2];
        float z11 = src[(ys + 1) * (width*factor) * 4 + ((xs + 1) * 4) + 2];
        out.z = (z00 + z01 + z10 + z11) / 4;


        dst[y * width * 3 + (x * 3) + 0] = (unsigned char)out.x;
        dst[y * width * 3 + (x * 3) + 1] = (unsigned char)out.y;
        dst[y * width * 3 + (x * 3) + 2] = (unsigned char)out.z;
        // dst[y * width * 4 + (x * 4) + 3] = (unsigned char)0;

    // }
    // dst[y * width * 4 + (x * 4) + 0] = src[ys * (width*2) * 4 + (xs * 4) + 0];
    // dst[y * width * 4 + (x * 4) + 1] = src[ys * (width*2) * 4 + (xs * 4) + 1];
    // dst[y * width * 4 + (x * 4) + 2] = src[ys * unsigned char(width*2) * 4 + (xs * 4) + 2];

}

void Resize(const int height, const int width, float* src, unsigned char* dst,const int factor){
    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    
    grid.x = getGridDim (width, block.x);
    grid.y = getGridDim (height, block.y);

    resizeImg<<<grid, block>>>(height, width, dst, src, factor);
    cudaSafeCall(cudaGetLastError());
}

__global__ void sampleGraphGen(float* model_buffer, int count, float* sample_points, int* d_count)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int rows_mb, cols_mb;
    rows_mb = cols_mb = 3072;

    if (i >= cols_mb * rows_mb)
        return;
    if (i >= count)
        return;

    if((i % 5000) != 0)
        return;
    int j = (i / 5000) - 1;
    if (j < 1024)
    {

        sample_points[j] = model_buffer[i];
        sample_points[j + 1] = model_buffer[i + rows_mb*cols_mb];
        sample_points[j + 2] = model_buffer[i + 2*rows_mb*cols_mb];
        sample_points[j + 3] = model_buffer[i + 7*rows_mb*cols_mb];

        atomicAdd(d_count, 1);

    }

}

void SampleGraph(DeviceArray<float>& model_buffer, int count, DeviceArray<float>& sample_points, int* h_count)
{

    int blocksize = 32*8;
    int numblocks = (count + blocksize - 1)/ blocksize;

    int *d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice);

    sampleGraphGen<<<numblocks, blocksize>>>(model_buffer, count, sample_points, d_count);

    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();


}