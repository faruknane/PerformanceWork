#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ErrorChecking.cpp"

//length res = length a 
//length a >= lenght b

template<typename T>
__global__ void AddVector_Kernel(T* res, T* a, T* b, size_t lengtha, size_t lengthb, T cofa, T cofb, T cofadd)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a[i] * cofa + b[i % lengthb] * cofb + cofadd;
}

extern "C" __declspec(dllexport) void AddFloat32(float* res, float* a, float* b, size_t lengtha, size_t lengthb, float cofa, float cofb, float cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    AddVector_Kernel<float> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddFloat64(double* res, double* a, double* b, size_t lengtha, size_t lengthb, double cofa, double cofb, double cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    AddVector_Kernel<double> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddInt16(short* res, short* a, short* b, size_t lengtha, size_t lengthb, short cofa, short cofb, short cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    AddVector_Kernel<short> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddInt32(int* res, int* a, int* b, size_t lengtha, size_t lengthb, int cofa, int cofb, int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    AddVector_Kernel<int> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AddInt64(long long int* res, long long int* a, long long int* b, size_t lengtha, size_t lengthb, long long int cofa, long long int cofb, long long int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    AddVector_Kernel<long long int> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofa, cofb, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}




template<typename T>
__global__ void AssignVector_Kernel(T* res, T* a, size_t length, size_t lenghta, T alpha, T beta)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
        res[i] = a[i % lenghta] * alpha + beta;
}

extern "C" __declspec(dllexport) void AssignFloat32(float* res, float* a, size_t length, size_t lengtha, float alpha, float beta)
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
    AssignVector_Kernel<float> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, lengtha, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignFloat64(double* res, double* a, size_t length, size_t lengtha, double alpha, double beta)
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
    AssignVector_Kernel<double> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, lengtha, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt16(short* res, short* a, size_t length, size_t lengtha, short alpha, short beta)
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
    AssignVector_Kernel<short> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, lengtha, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt32(int* res, int* a, size_t length, size_t lengtha, int alpha, int beta)
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
    AssignVector_Kernel<int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, lengtha, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void AssignInt64(long long int* res, long long int* a, size_t length, size_t lengtha, long long int alpha, long long int beta)
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
    AssignVector_Kernel<long long int> << <(length + th - 1) / th, th, 0, stream >> > (res, a, length, lengtha, alpha, beta);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}




template<typename T>
__global__ void MultiplyVector_Kernel(T* res, T* a, T* b, size_t lengtha, size_t lengthb, T cofmul, T cofadd)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a[i] * b[i % lengthb] * cofmul + cofadd;
}

extern "C" __declspec(dllexport) void MultiplyFloat32(float* res, float* a, float* b, size_t lengtha, size_t lengthb, float cofmul, float cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    MultiplyVector_Kernel<float> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void MultiplyFloat64(double* res, double* a, double* b, size_t lengtha, size_t lengthb, double cofmul, double cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    MultiplyVector_Kernel<double> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void MultiplyInt16(short* res, short* a, short* b, size_t lengtha, size_t lengthb, short cofmul, short cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    MultiplyVector_Kernel<short> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void MultiplyInt32(int* res, int* a, int* b, size_t lengtha, size_t lengthb, int cofmul, int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    MultiplyVector_Kernel<int> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void MultiplyInt64(long long int* res, long long int* a, long long int* b, size_t lengtha, size_t lengthb, long long int cofmul, long long int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    MultiplyVector_Kernel<long long int> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}




template<typename T>
__global__ void DivideVector_Kernel(T* res, T* a, T* b, size_t lengtha, size_t lengthb, T cofmul, T cofadd)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a[i] / b[i % lengthb] * cofmul + cofadd;
}

extern "C" __declspec(dllexport) void DivideFloat32(float* res, float* a, float* b, size_t lengtha, size_t lengthb, float cofmul, float cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    DivideVector_Kernel<float> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void DivideFloat64(double* res, double* a, double* b, size_t lengtha, size_t lengthb, double cofmul, double cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    DivideVector_Kernel<double> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void DivideInt16(short* res, short* a, short* b, size_t lengtha, size_t lengthb, short cofmul, short cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    DivideVector_Kernel<short> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void DivideInt32(int* res, int* a, int* b, size_t lengtha, size_t lengthb, int cofmul, int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    DivideVector_Kernel<int> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}

extern "C" __declspec(dllexport) void DivideInt64(long long int* res, long long int* a, long long int* b, size_t lengtha, size_t lengthb, long long int cofmul, long long int cofadd)
{
    int devid;
    CheckCudaError(cudaGetDevice(&devid), "GetDevice");
    cudaDeviceProp prop;
    CheckCudaError(cudaGetDeviceProperties(&prop, devid), "GetDeviceProp");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t th = prop.maxThreadsPerBlock;
    if (lengtha < th)
        th = lengtha;
    DivideVector_Kernel<long long int> << <(lengtha + th - 1) / th, th, 0, stream >> > (res, a, b, lengtha, lengthb, cofmul, cofadd);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    CheckCudaError(cudaStreamDestroy(stream), "StreamDestroy");
}


