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
               int Order, int TransA, int TransB, long M, long N, long K,
               float alpha, float* A, long lda, float* B, long ldb,
               float beta, float* C, long ldc);

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void* MKL_malloc(long size, int alignment);

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void MKL_free(void* a_ptr);


        [SuppressUnmanagedCodeSecurityAttribute]

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false)]
        public static extern void* vsExp(long n, float* x, float* y);

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurityAttribute]
        public static extern void* vmsExp(long n, float* x, float* y, long m);


        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                        ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void* vmsAdd(long n, float* a, float* b, float* y, long m);

        [SuppressUnmanagedCodeSecurityAttribute]

        [DllImport("MKL\\mkl_rt.dll", CallingConvention = CallingConvention.Cdecl,
                     ExactSpelling = true, SetLastError = false)]
        public static extern void* vmsMul(long n, float* a, float* b, float* y, long m);

  

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
