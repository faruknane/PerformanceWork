#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ErrorChecking.cpp"


template<typename T>
__global__ void AddVector_Kernel1(T* res, T* a, T* b, int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length)
        res[i] = a[i] + b[i];
}

extern "C" __declspec(dllexport) void AddFloat32(float* res, float* a, float* b, int length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<float> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddFloat64(double* res, double* a, double* b, int length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<double> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}