#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
//#include <cudnn.h>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ErrorChecking.cpp"
#include <sstream>
#include <iostream>

using namespace std;

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    abort();														   \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


extern "C" __declspec(dllexport) void NSetDevice(int gpuid)
{
	CheckCudaError(cudaSetDevice(gpuid), "setdevice");
}

extern "C" __declspec(dllexport) int NGetDevice()
{
	int devid;
	CheckCudaError(cudaGetDevice(&devid), "getdevice");
	return devid;
}

extern "C" __declspec(dllexport) void* NAllocate(size_t bytesize, int gpuid)
{
	NSetDevice(gpuid);
	void* a = 0;
	CheckCudaError(cudaMalloc((void**)&a, bytesize), "cudamalloc");
	if (a == 0)
		FatalError("Allocation Error on GPU device!");
	return a;
}

extern "C" __declspec(dllexport) void NFree(void* arr)
{
	CheckCudaError(cudaFree(arr), "Free");
}

extern "C" __declspec(dllexport) void NCopyArray(void* src, void* dst, size_t bytesize)
{
	CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDefault), "memcpy");
}


const size_t arraySize = 10000000;
const float a[arraySize] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.1f };
const float b[arraySize] = { 0, 0, 0, 0, 0 };
float c[arraySize] = { 1,2,3,4,5 };

__global__ void addSingleArrays2Kernel(const float* a, const float* b, float* c, size_t size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		c[i] = a[i] + b[i];
}



int main()
{
	NSetDevice(0);
	int devid = NGetDevice();
	float* dev_a = (float*)NAllocate(arraySize * sizeof(float), devid);
	float* dev_b = (float*)NAllocate(arraySize * sizeof(float), devid);
	float* dev_c = (float*)NAllocate(arraySize * sizeof(float), devid);
	
	std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();
	NCopyArray((void*)a, dev_a, arraySize * sizeof(float));
	NCopyArray((void*)b, dev_b, arraySize * sizeof(float));
	cout << "bitti1" << endl;
	std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
	std::cout << "Time difference1 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - begin3).count() << "ms" << std::endl;

	
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	for (int i = 0; i < 1; i++)
	{
		cudaStreamSynchronize(stream1);
	}
	cout << "bitti2" << endl;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference2 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;


	std::chrono::steady_clock::time_point begin6 = std::chrono::steady_clock::now();
	for (int i = 0; i < 1; i++)
	{
		cudaStreamSynchronize(stream1);
	}
	cout << "bitti6" << endl;
	std::chrono::steady_clock::time_point end6 = std::chrono::steady_clock::now();
	std::cout << "Time difference6 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end6 - begin6).count() << "ms" << std::endl;



	std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
	NCopyArray(dev_c, c, arraySize * sizeof(float));
	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
		c[0], c[1], c[2], c[3], c[4]);
	
	std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	std::cout << "Time difference4 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "ms" << std::endl;



	std::chrono::steady_clock::time_point begin4 = std::chrono::steady_clock::now();
	NFree(dev_a);
	NFree(dev_b);
	NFree(dev_c);
	std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
	std::cout << "Time difference5 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end4 - begin4).count() << "ms" << std::endl;


	auto cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}