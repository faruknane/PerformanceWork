#include <cublasLt.h>
#include "ErrorChecking.cpp"

//https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
/// Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasSgemm,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
void LtSgemm(cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    size_t m,
    size_t n,
    size_t k,
    const float* alpha, /* host pointer */
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    const float* beta, /* host pointer */
    float* C,
    size_t ldc,
    float* D,
    size_t ldd,
    void* workspace,
    size_t workspaceSize, cudaStream_t* s) {


    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CheckCublasError2(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CheckCublasError2(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CheckCublasError2(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CheckCublasError2(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CheckCublasError2(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CheckCublasError2(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
    CheckCublasError2(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, ldd));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CheckCublasError2(cublasLtMatmulPreferenceCreate(&preference));
    CheckCublasError2(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    CheckCublasError2(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CheckCublasError2(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CheckCublasError2(cublasLtMatmul(ltHandle,
        operationDesc,
        alpha,
        A,
        Adesc,
        B,
        Bdesc,
        beta,
        C,
        Cdesc,
        D,
        Ddesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        *s));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) CheckCublasError2(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CheckCublasError2(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Ddesc) CheckCublasError2(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Bdesc) CheckCublasError2(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CheckCublasError2(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CheckCublasError2(cublasLtMatmulDescDestroy(operationDesc));
}

void MatrixMultiplyFloat32(float* A, float* B, float* C, size_t dim1, size_t dim2, size_t dim3, float alpha, float beta)
{

    cublasLtHandle_t ltHandle;
    cudaStream_t stream;
    CheckCudaError2(cudaStreamCreate(&stream));

    CheckCublasError2(cublasLtCreate(&ltHandle));

    LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim3, dim1, dim2,
        &alpha, B,
        dim3, A, dim2, &beta, C, dim3, C, dim3, nullptr, 0, &stream);

    CheckCudaError(cudaStreamSynchronize(stream), "MatrixMultiplyFloat32 Stream Sync\n");

    CheckCublasError2(cublasLtDestroy(ltHandle));
}