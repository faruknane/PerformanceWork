/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <cutensor/types.h>
#include "ErrorChecking.cpp"

using namespace std;

/* This routine computes the tensor contraction \f[ D = alpha * A * B + beta * C \f] using the staged-API */
cutensorStatus_t cutensorContractionSimple(const cutensorHandle_t* handle,
    const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
    const void* B, const cutensorTensorDescriptor_t* descB, const int32_t modeB[],
    const void* beta, const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[],
    void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
    cutensorComputeType_t typeCompute, cutensorAlgo_t algo, cutensorWorksizePreference_t workPref,
    cudaStream_t stream)
{
    /**********************************************
     * Retrieve the memory alignment for each tensor
     **********************************************/

    uint32_t alignmentRequirementA;
    CheckCutensorError(cutensorGetAlignmentRequirement(handle,
        A, descA, &alignmentRequirementA));

    uint32_t alignmentRequirementB;
    CheckCutensorError(cutensorGetAlignmentRequirement(handle,
        B, descB, &alignmentRequirementB));

    uint32_t alignmentRequirementC;
    CheckCutensorError(cutensorGetAlignmentRequirement(handle,
        C, descC, &alignmentRequirementC));

    uint32_t alignmentRequirementD;
    CheckCutensorError(cutensorGetAlignmentRequirement(handle,
        D, descD, &alignmentRequirementD));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorContractionDescriptor_t desc;
    CheckCutensorError(cutensorInitContractionDescriptor(handle,
        &desc,
        descA, modeA, alignmentRequirementA,
        descB, modeB, alignmentRequirementB,
        descC, modeC, alignmentRequirementC,
        descD, modeD, alignmentRequirementD,
        typeCompute));

    /**************************
    * Set the algorithm to use
    ***************************/

    cutensorContractionFind_t find;
    CheckCutensorError(cutensorInitContractionFind(
        handle, &find,
        CUTENSOR_ALGO_DEFAULT));

    /**********************
     * Query workspace
     **********************/

    size_t worksize = 0;
    CheckCutensorError(cutensorContractionGetWorkspace(handle,
        &desc,
        &find,
        workPref, &worksize));

    void* work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    }

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorContractionPlan_t plan;
    CheckCutensorError(cutensorInitContractionPlan(handle,
        &plan,
        &desc,
        &find,
        worksize));

    /**********************
     * Run
     **********************/

    CheckCutensorError(cutensorContraction(handle,
        &plan,
        alpha, A, B,
        beta, C, D,
        work, worksize, stream));

    return CUTENSOR_STATUS_SUCCESS;
}


extern "C" __declspec(dllexport) int Einsum(void* A_d, int nmodeA, int* modeA, int64_t * extentA, int typeAA,
    void* B_d, int nmodeB, int* modeB, int64_t * extentB, int typeBB,
    void* C_d, int nmodeC, int* modeC, int64_t * extentC, int typeCC,
    void* D_d, int nmodeD, int* modeD, int64_t * extentD, int typeDD,
    void* alpha, void* beta, int typeCompute2)
{
    cudaDataType_t typeA = (cudaDataType_t)typeAA;
    cudaDataType_t typeB = (cudaDataType_t)typeBB;
    cudaDataType_t typeC = (cudaDataType_t)typeCC;
    cudaDataType_t typeD = (cudaDataType_t)typeDD;
    cutensorComputeType_t typeCompute = (cutensorComputeType_t)typeCompute2;

    /*************************
     * cuTENSOR
     *************************/

    cutensorHandle_t handle;
    CheckCutensorError(cutensorInit(&handle));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    CheckCutensorError(cutensorInitTensorDescriptor(&handle,
        &descA,
        nmodeA,
        extentA,
        NULL /* stride */,
        typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    CheckCutensorError(cutensorInitTensorDescriptor(&handle,
        &descB,
        nmodeB,
        extentB,
        NULL /* stride */,
        typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    CheckCutensorError(cutensorInitTensorDescriptor(&handle,
        &descC,
        nmodeC,
        extentC,
        NULL /* stride */,
        typeC, CUTENSOR_OP_IDENTITY));


    CheckCutensorError(cutensorContractionSimple(&handle,
        alpha, A_d, &descA, modeA,
        B_d, &descB, modeB,
        beta, C_d, &descC, modeC,
        D_d, &descC, modeC,
        typeCompute, CUTENSOR_ALGO_DEFAULT,
        CUTENSOR_WORKSPACE_RECOMMENDED, stream /* stream */));


    CheckCudaError2(cudaStreamSynchronize(stream));
    CheckCudaError2(cudaStreamDestroy(stream));

    //todo destroy handle
    return 0;
}
