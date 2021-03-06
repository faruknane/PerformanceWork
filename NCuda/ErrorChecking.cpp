#define CheckCublasError(x, err)                                        \
    if ((x) != CUBLAS_STATUS_SUCCESS)                                   \
    {                                                                   \
        fprintf(stderr, "Cublas: ");                                    \
        fprintf(stderr, err);                                           \
        fprintf(stderr, " error!\n");                                   \
        exit(-1);                                                       \
    }                                                                     

#define CheckCublasError2(x)                                            \
    if ((x) != CUBLAS_STATUS_SUCCESS)                                   \
    {                                                                   \
        fprintf(stderr, "Cublas error!\n");                             \
        exit(-1);                                                       \
    }                                                                    

#define CheckCudaError(x, err)                                          \
    if ((x) != 0)                                                       \
    {                                                                   \
        fprintf(stderr, "Cuda: ");                                      \
        fprintf(stderr, err);                                           \
        fprintf(stderr, " error!\n");                                   \
        exit(-1);                                                       \
    }  

#define CheckCudaError2(x)                                              \
    if ((x) != 0)                                                       \
    {                                                                   \
        fprintf(stderr, "Cuda error!\n");                               \
        exit(-1);                                                       \
    }  
