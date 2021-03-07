#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "ErrorChecking.cpp"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>


extern "C" __declspec(dllexport) void MatrixMultiplyFloat32_Handle(void* handle, float* A, float* B, float* C, size_t dim1, size_t dim2, size_t dim3, float alpha, float beta)
{
    CheckCublasError(cublasSgemm(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, dim3, dim1, dim2,
        &alpha, B,
        dim3, A, dim2, &beta, C, dim3), "cublasSgemm");
}

extern "C" __declspec(dllexport) void MatrixMultiplyFloat32(float* A, float* B, float* C, size_t dim1, size_t dim2, size_t dim3, float alpha, float beta)
{
    cublasHandle_t handle;

    CheckCublasError(cublasCreate(&handle), "cublasCreate handle");

    MatrixMultiplyFloat32_Handle(&handle, A, B, C, dim1, dim2, dim3, alpha, beta);

    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    CheckCudaError(cudaStreamSynchronize(stream), "StreamSync");
    cublasDestroy(handle);
}
