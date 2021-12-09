#include "utils.h"

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)

void getDefaultSecurityDescriptor(CUmemAllocationProp *prop) {
#if defined(__linux__)
    return;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
    static OBJECT_ATTRIBUTES objAttributes;
    static bool objAttributesConfigured = false;

    if (!objAttributesConfigured) {
        PSECURITY_DESCRIPTOR secDesc;
        BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(
            sddl, SDDL_REVISION_1, &secDesc, NULL);
        if (result == 0) {
            printf("IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n",
                GetLastError());
        }

        InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);

        objAttributesConfigured = true;
    }

    prop->win32HandleMetaData = &objAttributes;
    return;
#endif
}

void CudaOps::allocateMem(size_t outSize){

    int deviceCount;
    int cudaDevice = cudaInvalidDeviceId;
    cudaSafeCall(cudaGetDeviceCount(&deviceCount));
    // need to fix
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp devProp = { };
        cudaGetDeviceProperties(&devProp, dev);
        if (true) {
            cudaDevice = dev;
            break;
        }

        // if (isVkPhysicalDeviceUuid(&devProp.uuid)) {
        //     cudaDevice = dev;
        //     break;
        // }
    }
    // if (cudaDevice == cudaInvalidDeviceId) {
    //     throw std::runtime_error("No Suitable device found!");
    // }
    size_t granularity = 0;
    cudaSetDevice(cudaDevice);
    CUmemGenericAllocationHandle cudaImgHandle;

    CUmemAllocationProp allocProp = { };
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = cudaDevice;
    allocProp.win32HandleMetaData = NULL;
    allocProp.requestedHandleTypes = ipcHandleTypeFlag;

    getDefaultSecurityDescriptor(&allocProp);

    cuMemGetAllocationGranularity(&granularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    imgSize = ROUND_UP_TO_GRANULARITY(outSize, granularity);
    cuMemAddressReserve(&d_ptr, imgSize, granularity, 0U, 0);
    cuMemCreate(&cudaImgHandle, imgSize, &allocProp, 0);
    cuMemExportToShareableHandle((void *)&imgShareableHandle, cudaImgHandle, ipcHandleTypeFlag, 0);



    cuMemMap(d_ptr, imgSize, 0, cudaImgHandle, 0);
    cuMemRelease(cudaImgHandle);

    d_output = (float*)d_ptr;
    CUmemAccessDesc accessDescriptor = {};
    accessDescriptor.location.id = cudaDevice;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Apply the access descriptor to the whole VA range. Essentially enables Read-Write access to the range.
    cuMemSetAccess(d_ptr, imgSize, &accessDescriptor, 1);


}

void CudaOps::cleanAllocations(){
    
        cuMemUnmap((CUdeviceptr)d_output, imgSize);
        cuMemAddressFree((CUdeviceptr)d_output, imgSize);
        close(imgShareableHandle);
        d_output = nullptr;
}

texture<uchar4, 2, cudaReadModeElementType> inTex;
__global__ void imageResizeKernel(float* dst, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    float scale_w = 1.178;
    float scale_h = 0.873;

    int xs = x * scale_h;
    int ys = y * scale_w;

    int i = width * y + x;

    float mean_r = 123.68f;
    float mean_g = 116.78f;
    float mean_b = 103.94f;

    float norm_r = 58.40f;
    float norm_g = 57.12f;
    float norm_b = 57.38f;


    uchar4 src = tex2D(inTex, xs, ys);



    // dst[y * width * 3 + (x * 3) + 0] = ((float)src.z - mean_r)/norm_r;
    // dst[y * width * 3 + (x * 3) + 1] = ((float)src.y - mean_b)/norm_b;
    // dst[y * width * 3 + (x * 3) + 2] = ((float)src.x - mean_g)/norm_g;

    dst[i] = ((float)src.z - mean_r)/norm_r;
    dst[i + (width * height)] = ((float)src.y - mean_b)/norm_b;
    dst[i + 2 * (width * height)] = ((float)src.x - mean_g)/norm_g;

    // int value = (float)src.x * 0.114f + (float)src.y * 0.299f + (float)src.z * 0.587f;

    // dst.ptr (y)[x] = value;
}

int CudaOps::imageResize(cudaArray * cuArr, int width, int height)
{
    allocateMem(width * height * 3 * (sizeof(float)));
    dim3 block (32, 8);
    dim3 grid (getGridDim (width, block.x), getGridDim (height, block.y));

    cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));

    imageResizeKernel<<<grid, block>>>(d_output, width, height);

    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(inTex));
    return imgShareableHandle;
};

__global__ void resizeImg(const int height, const int width, unsigned char * dst, float * src, const int scale_w, const int scale_h)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    // const float qnan = __int_as_float(0x7fffffff);

    int xs = x * scale_h;
    int ys = y * scale_w;

    float x00 = src[(ys + 0) * (width*scale_w) * 4 + ((xs + 0) * 4) + 0];
    float x01 = src[(ys + 0) * (width*scale_w) * 4 + ((xs + 1) * 4) + 0];
    float x10 = src[(ys + 1) * (width*scale_w) * 4 + ((xs + 0) * 4) + 0];
    float x11 = src[(ys + 1) * (width*scale_w) * 4 + ((xs + 1) * 4) + 0];

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

        float y00 = src[(ys + 0) * (width*scale_w) * 4 + ((xs + 0) * 4) + 1];
        float y01 = src[(ys + 0) * (width*scale_w) * 4 + ((xs + 1) * 4) + 1];
        float y10 = src[(ys + 1) * (width*scale_w) * 4 + ((xs + 0) * 4) + 1];
        float y11 = src[(ys + 1) * (width*scale_w) * 4 + ((xs + 1) * 4) + 1];
        out.y = (y00 + y01 + y10 + y11) / 4;


        float z00 = src[(ys + 0) * (width*2) * 4 + ((xs + 0) * 4) + 2];
        float z01 = src[(ys + 0) * (width*2) * 4 + ((xs + 1) * 4) + 2];
        float z10 = src[(ys + 1) * (width*2) * 4 + ((xs + 0) * 4) + 2];
        float z11 = src[(ys + 1) * (width*2) * 4 + ((xs + 1) * 4) + 2];
        out.z = (z00 + z01 + z10 + z11) / 4;


        dst[y * width * 3 + (x * 3) + 0] = (unsigned char)out.x;
        dst[y * width * 3 + (x * 3) + 1] = (unsigned char)out.y;
        dst[y * width * 3 + (x * 3) + 2] = (unsigned char)out.z;
        // dst[y * width * 3 + (x * 3) + 3] = (unsigned char)0;

    // }
    // dst[y * width * 4 + (x * 4) + 0] = src[ys * (width*2) * 4 + (xs * 4) + 0];
    // dst[y * width * 4 + (x * 4) + 1] = src[ys * (width*2) * 4 + (xs * 4) + 1];
    // dst[y * width * 4 + (x * 4) + 2] = src[ys * unsigned char(width*2) * 4 + (xs * 4) + 2];

}