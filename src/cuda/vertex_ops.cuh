#ifndef CUDA_CUDAFUNCS_CUH_
#define CUDA_CUDAFUNCS_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif


void test(void);
void test_wrapper(void);


#endif /* CUDA_CUDAFUNCS_CUH_ */
