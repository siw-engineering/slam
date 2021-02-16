/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"

__global__ void pyrDownGaussKernel (const PtrStepSz<float> src, PtrStepSz<float> dst, float sigma_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr (2 * y)[2 * x];

    int x_mi = max(0, 2*x - D/2) - 2*x;
    int y_mi = max(0, 2*y - D/2) - 2*y;

    int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
    int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;

    float sum = 0;
    float wall = 0;

    float weights[] = {0.375f, 0.25f, 0.0625f} ;

    for(int yi = y_mi; yi < y_ma; ++yi)
        for(int xi = x_mi; xi < x_ma; ++xi)
        {
            int val = src.ptr (2*y + yi)[2*x + xi];

            if (abs (val - center) < 3 * sigma_color)
            {
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
            }
        }


    dst.ptr (y)[x] = static_cast<int>(sum / wall);
}

void pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float sigma_color = 30;

    pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
    cudaCheckError();
}

// Generate a vertex map 'vmap' based on the depth map 'depth' and camera parameters
__global__ void computeVmapKernel(const PtrStepSz<float> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy, float depthCutoff)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < depth.cols && v < depth.rows)
    {
        float z = depth.ptr(v)[u] /*/ 1000.f*/; // load and convert: mm -> meters

        if(z != 0 && z < depthCutoff /*&& m == maskID*/) //FIXME
        {
            float vx = z * (u - cx) * fx_inv;
            float vy = z * (v - cy) * fy_inv;
            float vz = z;

            vmap.ptr (v                 )[u] = vx;
            vmap.ptr (v + depth.rows    )[u] = vy;
            vmap.ptr (v + depth.rows * 2)[u] = vz;
        }
        else
        {
            vmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        }
    }
}

void createVMap(const CameraModel& intr, const DeviceArray2D<float> & depth, DeviceArray2D<float> & vmap, const float depthCutoff)
{
    vmap.create (depth.rows () * 3, depth.cols ());

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (depth.cols (), block.x);
    grid.y = getGridDim (depth.rows (), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff);
    cudaSafeCall(cudaGetLastError());
}

__device__ float getRadius(float fx, float fy, float depth, float norm_z)
{
    float meanFocal = ((1.0 / abs(fx)) + (1.0 / abs(fy))) / 2.0;
    
    const float sqrt2 = 1.41421356237f;
    
    float radius = (depth / meanFocal) * sqrt2;

    float radius_n = radius;

    radius_n = radius_n / abs(norm_z);

    radius_n = min(2.0f * radius, radius_n);

    return radius_n;
}

__device__ float encodeColor(float3 c)
{
    int rgb = 0;
    rgb = int(round(c.x * 255.0f));
    rgb = (rgb << 8) + int(round(c.y * 255.0f));
    rgb = (rgb << 8) + int(round(c.z * 255.0f));
    return  (float)rgb;
}

__device__ float3 decodeColor(float c)
{
    float3 col;
    col.x = float(int(c) >> 16 & 0xFF) / 255.0f;
    col.y = float(int(c) >> 8 & 0xFF) / 255.0f;
    col.z = float(int(c) & 0xFF) / 255.0f;
    return col;
}


__device__ float confidence(float cx, float cy, float x, float y, float weighting)
{
    const float maxRadDist = 400; //sqrt((width * 0.5)^2 + (height * 0.5)^2)
    const float twoSigmaSquared = 0.72; //2*(0.6^2) from paper
    
    float3 pixelPosCentered = make_float3(x-cx, y-cy, 0);
    // vec2 pixelPosCentered = vec2(x, y) - cam.xy;
    float radialDist = sqrt(dot(pixelPosCentered, pixelPosCentered)) / maxRadDist;
    return exp((-(radialDist * radialDist) / twoSigmaSquared)) * weighting;
}


__global__ void initModelBufferKernel(float cx, float cy, float fx, float fy, float max_depth, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, const float* rgb, float* model_buffer, int rows, int cols, int* count)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    int vz = vmap.ptr(v + rows*2)[u];
    if ((vz <= 0) || (vz > max_depth))
    {
        return;
    }

    atomicAdd(count, 1);

    // replace this, hardcoding temporarily
    int rows_mb, cols_mb;
    rows_mb = cols_mb = 3072;

    int i = rows*v + u;

    //writing vertex and confidence
    model_buffer[i] = vmap.ptr(v)[u];
    model_buffer[i+ rows_mb*cols_mb] = vmap.ptr(v + rows)[u];
    model_buffer[i+2*rows_mb*cols_mb] = vz;
    model_buffer[i+3*rows_mb*cols_mb] = confidence(cx, cy, u, v, 1);

    // color encoding
    float3 c;
    float ec ;
    c.x = rgb[i];
    c.y = rgb[i + rows*cols];
    c.z = rgb[i + 2*rows*cols];
    ec = encodeColor(c);

    //writing color and time
    model_buffer[i+4*rows_mb*cols_mb] = ec; //x
    model_buffer[i+5*rows_mb*cols_mb] = 0;//y
    model_buffer[i+6*rows_mb*cols_mb] = 1;//z
    model_buffer[i+7*rows_mb*cols_mb] = 1;//w time

    //writing normals
    model_buffer[i+8*rows_mb*cols_mb] = nmap.ptr(v)[u];
    model_buffer[i+9*rows_mb*cols_mb] = nmap.ptr(v + rows)[u];
    model_buffer[i+10*rows_mb*cols_mb] = nmap.ptr(v + rows*2)[u];
    model_buffer[i+11*rows_mb*cols_mb] = getRadius(fx, fy, vmap.ptr(v + rows*2)[u], nmap.ptr(v + rows*2)[u]);

}

void initModelBuffer(const CameraModel& intr, const float depthCutOff, const DeviceArray2D<float> & vmap, const DeviceArray2D<float> & nmap, const DeviceArray<float> & rgb, DeviceArray<float> & model_buffer, int* h_count)
{
    int *d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice);

    int cols, rows;
    rows = vmap.rows()/3;
    cols = vmap.cols();
    dim3 block(32, 8);
    dim3 grid(getGridDim(cols, block.x), getGridDim(rows, block.y));

    initModelBufferKernel<<<grid, block>>>(intr.cx, intr.cy, intr.fx, intr.fy, depthCutOff, vmap, nmap, rgb, model_buffer, rows, cols, d_count);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
}

// __global__ void kernelCodeKernel(float *result)
// {
//     int index = threadIdx.x+blockIdx.x*blockDim.x;
//     atomicAdd(result, 1.0f);
    
// }
// void kernelCode(){

//     float h_result, *d_result;
//     cudaMalloc((void **)&d_result, sizeof(float));
//     h_result = 0.0f;
//     cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

//     int rows, cols;
//     rows = 640;
//     cols = 480;
//     dim3 block(32, 8);
//     dim3 grid(getGridDim(cols, block.x), getGridDim(rows, block.y));

//     kernelCodeKernel<<<grid, block>>>(d_result);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout<< "result = " << h_result << std::endl;
// }

__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1)
    {
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    float3 v00, v01, v10;
    v00.x = vmap.ptr (v  )[u];
    v01.x = vmap.ptr (v  )[u + 1];
    v10.x = vmap.ptr (v + 1)[u];

    if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
    {
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];
        v10.y = vmap.ptr (v + 1 + rows)[u];

        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;
    }
    else
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
    nmap.create (vmap.rows (), vmap.cols ());

    int rows = vmap.rows () / 3;
    int cols = vmap.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (cols, block.x);
    grid.y = getGridDim (rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall (cudaGetLastError ());
}

__global__ void tranformMapsKernel(int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                                   const mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
            vsrc.y = vmap_src.ptr (y + rows)[x];
            vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

            vdst = Rmat * vsrc + tvec;

            vmap_dst.ptr (y + rows)[x] = vdst.y;
            vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
        }

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
            nsrc.y = nmap_src.ptr (y + rows)[x];
            nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

            ndst = Rmat * nsrc;

            nmap_dst.ptr (y + rows)[x] = ndst.y;
            nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
    }
}

void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const mat33& Rmat, const float3& tvec,
                  DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_src.cols();
    int rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void copyMapsKernel(int rows, int cols, const float * vmap_src, const float * nmap_src,
                               PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
        vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
        vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            vdst = vsrc;
        }

        vmap_dst.ptr (y)[x] = vdst.x;
        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];
        nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];
        nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            ndst = nsrc;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
        nmap_dst.ptr (y + rows)[x] = ndst.y;
        nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
    }
}

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_dst.cols();
    int rows = vmap_dst.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}



__global__ void copyMapsKernel2D_2_2D(int rows, int cols, PtrStepSz<float> vmap_src, PtrStep<float> nmap_src,
                               PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        // vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = vmap_src.ptr(y)[x];
        vsrc.y = vmap_src.ptr(y+rows)[x];
        vsrc.z = vmap_src.ptr(y+2*rows)[x];

        if(!(vsrc.z == 0))
        {
            vdst = vsrc;
        }

        vmap_dst.ptr (y)[x] = vdst.x;
        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        nsrc.x = nmap_src.ptr(y)[x];
        nsrc.y = nmap_src.ptr(y+rows)[x];
        nsrc.z = nmap_src.ptr(y+2*rows)[x];

        if(!(vsrc.z == 0))
        {
            ndst = nsrc;
        }
        nmap_dst.ptr (y)[x] = ndst.x;
        nmap_dst.ptr (y + rows)[x] = ndst.y;
        nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
    }
}

void copyMaps(const DeviceArray2D<float>& vmap_src,
              const DeviceArray2D<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_dst.cols();
    int rows = vmap_dst.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyMapsKernel2D_2_2D<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void copyMapsKernel2D_2_1D(int rows, int cols, PtrStepSz<float> vmap_src, PtrStep<float> nmap_src,
                                 float * vmap_dst,   float * nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = vmap_src.ptr (y)[x];
        vsrc.y = vmap_src.ptr (y + rows)[x];
        vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

        if(!(vsrc.z == 0))
        {
            vdst = vsrc;
        }

        vmap_dst[y * cols * 4 + (x * 4) + 0] = vdst.x;
        vmap_dst[y * cols * 4 + (x * 4) + 1] = vdst.y;
        vmap_dst[y * cols * 4 + (x * 4) + 2] = vdst.z;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        nsrc.x = nmap_src.ptr (y)[x] ;
        nsrc.y = nmap_src.ptr (y + rows)[x] ;
        nsrc.z = nmap_src.ptr (y + 2 * rows)[x] ;

        if(!(vsrc.z == 0))
        {
            ndst = nsrc;
        }

        nmap_dst[y * cols * 4 + (x * 4) + 0]= ndst.x;
        nmap_dst[y * cols * 4 + (x * 4) + 1]= ndst.y;
        nmap_dst[y * cols * 4 + (x * 4) + 2]= ndst.z;
    }
}

void copyMaps(const DeviceArray2D<float>& vmap_src,
              const DeviceArray2D<float>& nmap_src,
              DeviceArray<float>& vmap_dst,
              DeviceArray<float>& nmap_dst)
{
    int cols = vmap_src.cols();
    int rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3 * cols);
    nmap_dst.create(rows * 3 * cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyMapsKernel2D_2_1D<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void copyDMapsKernel2D_2_2D(int rows, int cols, PtrStepSz<float> dmap_src, PtrStepSz<float> dmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        dmap_dst.ptr(y)[x] = dmap_src.ptr(y)[x];
    }
}

void copyDMaps(const DeviceArray2D<float>& dmap_src,
              DeviceArray2D<float>& dmap_dst)
{
    int cols = dmap_src.cols();
    int rows = dmap_src.rows();


    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyDMapsKernel2D_2_2D<<<grid, block>>>(rows, cols, dmap_src, dmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void pyrDownKernelGaussF(const PtrStepSz<float> src, PtrStepSz<float> dst, float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    float center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src.ptr (cy)[cx]))
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }
    dst.ptr (y)[x] = (float)(sum / (float)count);
}

template<bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows, const PtrStep<float> input, PtrStep<float> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output.ptr (y)[x] = qnan;
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

        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
    }
}

template<bool normalize>
void resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    int in_cols = input.cols ();
    int in_rows = input.rows () / 3;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create (out_rows * 3, out_cols);

    dim3 block (32, 8);
    dim3 grid (getGridDim (out_cols, block.x), getGridDim (out_rows, block.y));
    resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<false>(input, output);
}

void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<true>(input, output);
}

//FIXME Remove
/*
void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time);
}*/

//FIXME Remove
/*
__global__ void testKernel(cudaSurfaceObject_t tex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 960 || y >= 540)
        return;

    / *
    const int D = 5;

    float center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src.ptr (cy)[cx]))
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }* /
    //dst.ptr (y)[x] = (float)(sum / (float)count);
    //data[y * 960 + x] = x / 960.0;
    //data[8] = 0.4;
    float1 test = make_float1(0.99);
    surf2Dwrite(test, tex, x*sizeof(float1), y);
}

//FIXME Remove
void testCuda(cudaSurfaceObject_t surface)//(float* data)
{
    //dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (960, block.x), getGridDim (540, block.y));

    testKernel<<<grid, block>>>(surface);
    cudaCheckError();
}*/

void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};

    float * gauss_cuda;

    cudaSafeCall(cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25));
    cudaSafeCall(cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice));

    pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
    cudaCheckError();

    cudaFree(gauss_cuda);
}

__global__ void pyrDownKernelIntensityGauss(const PtrStepSz<unsigned char> src, PtrStepSz<unsigned char> dst, float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            //This might not be right, but it stops incomplete model images from making up colors
            if(src.ptr (cy)[cx] > 0)
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    dst.ptr (y)[x] = (sum / (float)count);
}

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, DeviceArray2D<unsigned char> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};

    float * gauss_cuda;

    cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
    cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

    pyrDownKernelIntensityGauss<<<grid, block>>>(src, dst, gauss_cuda);
    cudaCheckError();

    cudaFree(gauss_cuda);
}

/*void pyrDown2(const DeviceArray2D<unsigned char> & src, DeviceArray2D<unsigned char> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    //pyrDownUcharGauss<<<grid, block>>>(src, dst);
    pyrDownUcharGauss()
    cudaCheckError();
}*/

__global__ void verticesToDepthKernel(const float * vmap_src, PtrStepSz<float> dst, float cutOff)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    float z = vmap_src[y * dst.cols * 4 + (x * 4) + 2];

    dst.ptr(y)[x] = z > cutOff || z <= 0 ? __int_as_float(0x7fffffff)/*CUDART_NAN_F*/ : z;
}

void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float> & dst, float cutOff)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    verticesToDepthKernel<<<grid, block>>>(vmap_src, dst, cutOff);
    cudaCheckError();
}

texture<uchar4, 2, cudaReadModeElementType> inTex;

__global__ void bgr2IntensityKernel(PtrStepSz<unsigned char> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    uchar4 src = tex2D(inTex, x, y);

    int value = (float)src.x * 0.114f + (float)src.y * 0.299f + (float)src.z * 0.587f;

    printf("%d\n", value);

    dst.ptr (y)[x] = value;
}

void imageBGRToIntensity(cudaArray * cuArr, DeviceArray2D<unsigned char> & dst)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));

    bgr2IntensityKernel<<<grid, block>>>(dst);

    cudaCheckError();

    cudaSafeCall(cudaUnbindTexture(inTex));
}

__global__ void bgr2IntensityKernelDM(int rows, int cols, float * rgb_src, PtrStepSz<unsigned char> rgb_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < cols && y < rows)
    {
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        vsrc.x = rgb_src[y * cols * 4 + (x * 4) + 0];
        vsrc.y = rgb_src[rows * cols * 4 + y * cols * 4 + (x * 4) + 1];
        vsrc.z = rgb_src[2 * rows * cols * 4 + y * cols * 4 + (x * 4) + 2];
        int value = (float)vsrc.x * 0.114f + (float)vsrc.y * 0.299f + (float)vsrc.z * 0.587f;
        rgb_dst.ptr(y)[x] = value;

    }
}

void imageBGRToIntensityDM(DeviceArray<float>& rgb_src, DeviceArray2D<unsigned char>& rgb_dst)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (rgb_dst.cols(), block.x), getGridDim (rgb_dst.rows(), block.y));
    int rows = rgb_dst.rows() / 3;
    int cols = rgb_dst.cols();
    bgr2IntensityKernelDM<<<grid, block>>>(rows, cols, rgb_src, rgb_dst);
    cudaCheckError();

}

__constant__ float gsobel_x3x3[9];
__constant__ float gsobel_y3x3[9];

__global__ void applyKernel(const PtrStepSz<unsigned char> src, PtrStep<short> dx, PtrStep<short> dy)
{

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= src.cols || y >= src.rows)
    return;

  float dxVal = 0;
  float dyVal = 0;

  int kernelIndex = 8;
  for(int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++)
  {
      for(int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++)
      {
          dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
          dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
          --kernelIndex;
      }
  }

  dx.ptr(y)[x] = dxVal;
  dy.ptr(y)[x] = dyVal;
}

void computeDerivativeImages(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy)
{
    static bool once = false;

    if(!once)
    {
        float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
                           0.79451, -0.00000, -0.79451,
                           0.52201,  0.00000, -0.52201};

        float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
                           0.00000, 0.00000, 0.00000,
                           -0.52201, -0.79451, -0.52201};

        cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
        cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        once = true;
    }

    dim3 block(32, 8);
    dim3 grid(getGridDim (src.cols (), block.x), getGridDim (src.rows (), block.y));

    applyKernel<<<grid, block>>>(src, dx, dy);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}


__global__ void projectPointsKernel(const PtrStepSz<float> depth,
                                    PtrStepSz<float3> cloud,
                                    const float invFx,
                                    const float invFy,
                                    const float cx,
                                    const float cy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];

    cloud.ptr(y)[x].x = (float)((x - cx) * z * invFx);
    cloud.ptr(y)[x].y = (float)((y - cy) * z * invFy);
    cloud.ptr(y)[x].z = z;
}

void projectToPointCloud(const DeviceArray2D<float> & depth,
                         const DeviceArray2D<float3> & cloud,
                         CameraModel & intrinsics,
                         const int & level)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (depth.cols (), block.x), getGridDim (depth.rows (), block.y));

    CameraModel intrinsicsLevel = intrinsics(level);

    projectPointsKernel<<<grid, block>>>(depth, cloud, 1.0f / intrinsicsLevel.fx, 1.0f / intrinsicsLevel.fy, intrinsicsLevel.cx, intrinsicsLevel.cy);
    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}

__device__ float3 projectPoint(float3 p, int rows, int cols, float cx, float cy, float fx, float fy, float maxDepth)
{
    float3 pt = make_float3(((((fx * p.x) / p.z) + cx) - (cols * 0.5)) / (cols * 0.5),
                ((((fy * p.y) / p.z) + cy) - (rows * 0.5)) / (rows * 0.5),
                p.z / maxDepth);
    return pt;
}

__global__ void splatDepthPredictKernel(int rows, int cols, float cx, float cy, float fx, float fy, float maxDepth, float* tinv, float* model_buffer, PtrStepSz<float> color_dst, PtrStepSz<float> vmap_dst, PtrStepSz<float> nmap_dst, PtrStepSz<unsigned int> time_dst)
{
    // int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int rows_mb, cols_mb;
    rows_mb = cols_mb = 3072;
    int vp_w, vp_h;
    vp_w = cols;
    vp_h = rows;
    
    // int i = x*y; //indexing wrong (solved when run 1d kernel)

    if (i >= cols_mb * rows_mb)
        return;
    if (model_buffer[i] == 0 || model_buffer[i + rows_mb*cols_mb] == 0 || model_buffer[i + 2*rows_mb*cols_mb] == 0) 
        return;

    float4 vsrc = make_float4(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    float4 nsrc = make_float4(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

    //reading vertex and conf
    vsrc.x = model_buffer[i];
    vsrc.y = model_buffer[i + rows_mb*cols_mb];
    vsrc.z = model_buffer[i + 2*rows_mb*cols_mb];
    vsrc.w = model_buffer[i + 3*rows_mb*cols_mb];

    //reading normal and radius
    nsrc.x = model_buffer[i+8*rows_mb*cols_mb];
    nsrc.y = model_buffer[i+9*rows_mb*cols_mb];
    nsrc.z = model_buffer[i+10*rows_mb*cols_mb];
    nsrc.w = model_buffer[i+11*rows_mb*cols_mb];

    //reading color
    float c;
    c = model_buffer[i+4*rows_mb*cols_mb]; //x

    //reading time
    unsigned int t;
    t = (unsigned int)model_buffer[i+7*rows_mb*cols_mb];


    if (isnan (vsrc.x) || isnan(vsrc.y) || isnan(vsrc.z))
        return;
    if (isnan (nsrc.x) || isnan(nsrc.y) || isnan(nsrc.z))
        return;

    float3 v_ = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    float3 n_ = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

    v_.x = tinv[0]*vsrc.x + tinv[1]*vsrc.y + tinv[2]*vsrc.z + tinv[3]*1;
    v_.y = tinv[4]*vsrc.x + tinv[5]*vsrc.y + tinv[6]*vsrc.z + tinv[7]*1;
    v_.z = tinv[8]*vsrc.x + tinv[9]*vsrc.y + tinv[10]*vsrc.z + tinv[11]*1;

    n_.x = tinv[0]*nsrc.x + tinv[1]*nsrc.y + tinv[2]*nsrc.z;
    n_.y = tinv[4]*nsrc.x + tinv[5]*nsrc.y + tinv[6]*nsrc.z;
    n_.z = tinv[8]*nsrc.x + tinv[9]*nsrc.y + tinv[10]*nsrc.z;
    n_ = normalized(n_);

    //to compute x,y cords (gl_fragcords)
    //TO DO need to normalize v_ 
    float3 fc;
    fc = projectPoint(v_, cx, cy, fx, fy, rows, cols, maxDepth);
    fc.x = fc.x * 0.5f + 0.5f; 
    fc.y = fc.y * 0.5f + 0.5f; 
    fc.x = fc.x * vp_w/2; // TO DO undo dividing by 2 when normlzing is fixed
    fc.y = fc.y * vp_h/2;

    int x, y;
    x = (int)fc.x;
    y = (int)fc.y;

    // printf("cords :%f\t%f\n", fc.x, fc.y);


    float3 l = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    float3 cp = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

    l.x = (fc.x - cx)/fx;
    l.y = (fc.y - cy)/fy;
    l.z = 1;
    l = normalized(l);

    float coeff;
    coeff = dot(v_, n_) / dot(l, n_);
    cp.x = l.x * coeff;
    cp.y = l.y * coeff;
    cp.z = l.z * coeff;

    float3 dc; 
    dc = decodeColor(c);

    //writing color
    color_dst.ptr(y)[x] = dc.x;
    color_dst.ptr(y + rows)[x] = dc.y;
    color_dst.ptr(y + rows  * 2)[x] = dc.z;
    color_dst.ptr(y + rows  * 3)[x] = 1;

    //writing vertex and conf
    vmap_dst.ptr(y)[x] = (x - cx)*cp.z*(1/fx);
    vmap_dst.ptr(y + rows)[x] = (y - cy)*cp.z*(1/fy);
    vmap_dst.ptr(y + rows * 2)[x] = cp.z;
    vmap_dst.ptr(y + rows * 3)[x] = vsrc.w;


    //writing normal and radius
    nmap_dst.ptr(y       )[x] = n_.x;
    nmap_dst.ptr(y + rows)[x] = n_.y;
    nmap_dst.ptr(y + 2 * rows)[x] = n_.z;
    nmap_dst.ptr(y + 3 * rows)[x] = nsrc.w;


    //writing time
    time_dst.ptr(y)[x] = t;

}

void splatDepthPredict(const CameraModel& intr, int rows, int cols,  float maxDepth, float* pose_inv, DeviceArray<float>& model_buffer, int count, DeviceArray2D<float>& color_dst, DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst, DeviceArray2D<unsigned int>& time_dst)
{
    // dim3 block (32, 8);
    // dim3 grid (getGridDim (cols, block.x), getGridDim (rows, block.y));
    // dim3 grid (getGridDim (cols, block.x), getGridDim (rows, block.y)); //change threads to count TO DO
    int blocksize = 32*8;
    int numblocks = (count + blocksize - 1)/ blocksize;

    //TO DO initialize device arrays with 0s
    color_dst.create(rows*4, cols);
    vmap_dst.create(rows*4, cols);
    nmap_dst.create(rows*4, cols);
    time_dst.create(rows, cols);


    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    float * tinv;
    cudaSafeCall(cudaMalloc((void**) &tinv, sizeof(float) * 16));
    cudaSafeCall(cudaMemcpy(tinv, pose_inv, sizeof(float) * 16, cudaMemcpyHostToDevice));

    splatDepthPredictKernel<<<numblocks, blocksize>>>(rows, cols, cx, cy , fx, fy, maxDepth, tinv, model_buffer, color_dst, vmap_dst, nmap_dst, time_dst);
    cudaCheckError();

}

__global__ void predictIndiciesKernel(float* model_buffer, int rows, int cols, float* tinv, float cx, float cy, float fx, float fy, float maxDepth, int time, int timeDelta, PtrStepSz<float> color_dst, PtrStepSz<float> vmap_pi, PtrStepSz<float> ct_pi, PtrStepSz<float> nmap_pi, PtrStepSz<unsigned int> index_pi)
{
    /* 
    solve modelbuffer threads problem
    */
    // int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int rows_mb, cols_mb;
    rows_mb = cols_mb = 3072;
    // int i = y* rows + x;
    float xu = 0;
    float yv = 0;

    if (i >= rows_mb*cols_mb)
        return;

    int vz = model_buffer[i + 2*rows_mb*cols_mb];
    int cw = model_buffer[i+7*rows_mb*cols_mb];
    int vertexId;
    float3 vsrc = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

    if ((vz < 0 ) || (vz > maxDepth) || (time - cw > timeDelta))
    {
        vsrc.x = -10;
        vsrc.y = -10;
        vertexId = 0;
    }
    else
    {
        float3 v_ = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        float3 nsrc = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        float3 n_ = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = model_buffer[i];
        vsrc.y = model_buffer[i + rows_mb*cols_mb];
        vsrc.z = model_buffer[i + 2*rows_mb*cols_mb];

        nsrc.x = model_buffer[i+8*rows_mb*cols_mb];
        nsrc.y = model_buffer[i+9*rows_mb*cols_mb];
        nsrc.z = model_buffer[i+10*rows_mb*cols_mb];

        v_.x = tinv[0]*vsrc.x + tinv[1]*vsrc.y + tinv[2]*vsrc.z + tinv[3]*1;
        v_.y = tinv[4]*vsrc.x + tinv[5]*vsrc.y + tinv[6]*vsrc.z + tinv[7]*1;
        v_.z = tinv[8]*vsrc.x + tinv[9]*vsrc.y + tinv[10]*vsrc.z + tinv[11]*1;

        xu = ((((fx* v_.x) / v_.z) + cx) - (cols * 0.5)) / (cols * 0.5);
        yv = ((((fy * v_.y) / v_.z) + cy) - (rows * 0.5)) / (rows * 0.5);
        // vertexId = gl_VertexID;
        vertexId = i;

        n_.x = tinv[0]*nsrc.x + tinv[1]*nsrc.y + tinv[2]*nsrc.z;
        n_.y = tinv[4]*nsrc.x + tinv[5]*nsrc.y + tinv[6]*nsrc.z;
        n_.z = tinv[8]*nsrc.x + tinv[9]*nsrc.y + tinv[10]*nsrc.z;
        n_ = normalized(n_);

        /*
        TO DO
        write outputs 
        gl_Position = vec4(x, y, vPosHome.z / maxDepth, 1.0);
        vPosition0 = vec4(vPosHome.xyz, vPosition.w);
        vColorTime0 = vColorTime;
        vNormRad0 = vec4(normalize(mat3(t_inv) * vNormRad.xyz), vNormRad.w);
        */
        vmap_pi.ptr(y)[x] = v_.x;
        vmap_pi.ptr(y + rows)[x] = v_.y;
        vmap_pi.ptr(y + rows * 2)[x] = v_.z;
        vmap_pi.ptr(y + rows * 3)[x] = model_buffer[i + 3*rows_mb*cols_mb];

        ct_pi.ptr(y)[x] = model_buffer[i+4*rows_mb*cols_mb];
        ct_pi.ptr(y + rows)[x] = model_buffer[i+5*rows_mb*cols_mb];
        ct_pi.ptr(y + rows * 2)[x] = model_buffer[i+6*rows_mb*cols_mb];
        ct_pi.ptr(y + rows * 3)[x] = model_buffer[i+7*rows_mb*cols_mb];

        nmap_pi.ptr(y)[x] = n_.x;
        nmap_pi.ptr(y + rows)[x] = n_.y;
        nmap_pi.ptr(y + rows * 2)[x] = n_.z;
        nmap_pi.ptr(y + rows * 3)[x] = model_buffer[i + 11*rows_mb*cols_mb];

        index_pi.ptr(y)[x] = i;

    }
}

void predictIndicies(const CameraModel& intr, int rows, int cols,  float* pose_inv, DeviceArray<float>& model_buffer, DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{

}

__global__ void fuseKernel()
{
    
}

void fuse()
{

}


