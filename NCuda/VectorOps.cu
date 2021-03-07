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
__global__ void AddVector_Kernel1(T* res, T* a, T* b, size_t length)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length)
        res[i] = a[i] + b[i];
}

extern "C" __declspec(dllexport) void AddFloat32(float* res, float* a, float* b, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<float> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddFloat64(double* res, double* a, double* b, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<double> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}



extern "C" __declspec(dllexport) void AddInt16(short* res, short* a, short* b, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<short> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


extern "C" __declspec(dllexport) void AddInt32(int* res, int* a, int* b, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


extern "C" __declspec(dllexport) void AddInt64(long long int* res, long long int* a, long long int* b, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel1<long long int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


template<typename T>
__global__ void AddVector_Kernel2(T* res, T* a, T* b, size_t length, T cofa, T cofb, T cofadd)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length)
        res[i] = a[i] * cofa + b[i] * cofb + cofadd;
}

extern "C" __declspec(dllexport) void AddFloat32_Coefficients(float* res, float* a, float* b, size_t length, float cofa, float cofb, float cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel2<float> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddFloat64_Coefficients(double* res, double* a, double* b, size_t length, double cofa, double cofb, double cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel2<double> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddInt16_Coefficients(short* res, short* a, short* b, size_t length, short cofa, short cofb, short cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel2<short> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddInt32_Coefficients(int* res, int* a, int* b, size_t length, int cofa, int cofb, int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel2<int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddInt64_Coefficients(long long int* res, long long int* a, long long int* b, size_t length, long long int cofa, long long int cofb, long long int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AddVector_Kernel2<long long int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, b, length, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


template<typename T>
__global__ void AssignVector_Kernel1(T* res, T* a, size_t length)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
        res[i] = a[i];
}

extern "C" __declspec(dllexport) void AssignFloat32(float* res, float* a, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel1<float> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignFloat64(double* res, double* a, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel1<double> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt16(short* res, short* a, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel1<short> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


extern "C" __declspec(dllexport) void AssignInt32(int* res, int* a, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel1<int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


extern "C" __declspec(dllexport) void AssignInt64(long long int* res, long long int* a, size_t length)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel1<long long int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

template<typename T>
__global__ void AssignVector_Kernel2(T* res, T* a, size_t length, T alpha, T beta)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
        res[i] = a[i] * alpha + beta;
}

extern "C" __declspec(dllexport) void AssignFloat32_Coefficients(float* res, float* a, size_t length, float alpha, float beta)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel2<float> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignFloat64_Coefficients(double* res, double* a, size_t length, double alpha, double beta)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel2<double> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt16_Coefficients(short* res, short* a, size_t length, short alpha, short beta)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel2<short> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt32_Coefficients(int* res, int* a, size_t length, int alpha, int beta)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel2<int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt64_Coefficients(long long int* res, long long int* a, size_t length, long long int alpha, long long int beta)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (length < th)
        th = length;
    AssignVector_Kernel2<long long int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}