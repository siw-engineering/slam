#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include "../cuda/convenience.cuh"
#include <unistd.h>

class CudaOps {

public:
	CudaOps(){};
	void allocateMem(size_t imgSize);
	int imageResize(cudaArray * cuArr, int width, int height);
	void cleanAllocations();
	// CU_MEM_HANDLE_TYPE_WIN32 for windows
	// CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR for linux
	CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	float *d_output = nullptr;
	CUdeviceptr d_ptr = 0U;
	size_t imgSize;
    typedef int ShareableHandle;
    ShareableHandle imgShareableHandle;

};