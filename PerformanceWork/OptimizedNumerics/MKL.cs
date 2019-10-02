using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public static unsafe class MKL
    {
        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void cblas_sgemm(
               int Order, int TransA, int TransB, int M, int N, int K,
               float alpha, float* A, int lda, float* B, int ldb,
               float beta, float* C, int ldc);

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void* MKL_malloc(int size, int alignment);

        [SuppressUnmanagedCodeSecurityAttribute]

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false)]
        public static extern void* vsExp(int n, float* x, float* y);

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurityAttribute]
        public static extern void* vmsExp(int n, float* x, float* y, long m);


        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void* vmsAdd(int n, float* a, float* b, float* y, long m);

        [SuppressUnmanagedCodeSecurityAttribute]

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                     ExactSpelling = true, SetLastError = false)]
        public static extern void* vmsMul(int n, float* a, float* b, float* y, long m);


        [SuppressUnmanagedCodeSecurityAttribute]

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false)]
        public static extern void mkl_simatcopy(char ordering, char trans, int rows, int cols, float alpha, float* AB, int lda, int ldb);


        [SuppressUnmanagedCodeSecurityAttribute]
        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false)]
        public static extern void pslacpy(char* uplo, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb);

        public sealed class ORDER
        {
            private ORDER() { }
            public static int RowMajor = 101;  /* row-major arrays */
            public static int ColMajor = 102;  /* column-major arrays */
        }

        /** Constants for CBLAS_TRANSPOSE enum, file "mkl_cblas.h" */
        public sealed class TRANSPOSE
        {
            private TRANSPOSE() { }
            public static int NoTrans = 111; /* trans='N' */
            public static int Trans = 112; /* trans='T' */
            public static int ConjTrans = 113; /* trans='C' */

        }
    }
}
