using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using static PerformanceWork.NCuda.CudaTypes;

namespace PerformanceWork.NCuda
{
    public static unsafe class CudaKernels
    {
        #region Add Vector Opreation
       
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat32(float* res, float* a, float* b, long lengtha, long lengthb, float cofa, float cofb, float cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat64(double* res, double* a, double* b, long lengtha, long lengthb, double cofa, double cofb, double cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt16(short* res, short* a, short* b, long lengtha, long lengthb, short cofa, short cofb, short cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt32(int* res, int* a, int* b, long lengtha, long lengthb, int cofa, int cofb, int cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt64(long* res, long* a, long* b, long lengtha, long lengthb, long cofa, long cofb, long cofadd);
        #endregion

        #region Assign Vector Operation
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat32(float* res, float* a, long lengthres, long lengtha, float alpha, float beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat64(double* res, double* a, long lengthres, long lengtha, double alpha, double beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt16(short* res, short* a, long lengthres, long lengtha, short alpha, short beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt32(int* res, int* a, long lengthres, long lengtha, int alpha, int beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt64(long* res, long* a, long lengthres, long lengtha, long alpha, long beta);
        #endregion

        #region Multiply Vector Opreation
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void MultiplyFloat32(float* res, float* a, float* b, long lengtha, long lengthb, float cofmul, float cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void MultiplyFloat64(double* res, double* a, double* b, long lengtha, long lengthb, double cofmul, double cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void MultiplyInt16(short* res, short* a, short* b, long lengtha, long lengthb, short cofmul, short cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void MultiplyInt32(int* res, int* a, int* b, long lengtha, long lengthb, int cofmul, int cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void MultiplyInt64(long* res, long* a, long* b, long lengtha, long lengthb, long cofmul, long cofadd);
        #endregion

        #region Divide Vector Opreation
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void DivideFloat32(float* res, float* a, float* b, long lengtha, long lengthb, float cofdiv, float cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void DivideFloat64(double* res, double* a, double* b, long lengtha, long lengthb, double cofdiv, double cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void DivideInt16(short* res, short* a, short* b, long lengtha, long lengthb, short cofdiv, short cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void DivideInt32(int* res, int* a, int* b, long lengtha, long lengthb, int cofdiv, int cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void DivideInt64(long* res, long* a, long* b, long lengtha, long lengthb, long cofdiv, long cofadd);
        #endregion

        #region Einsum Operation
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "Einsum"), SuppressUnmanagedCodeSecurity]
        public static extern int Einsum_Native(
            void* A_d, int nmodeA, int* modeA, long* extentA, long* strideA, int typeAA,
            void* B_d, int nmodeB, int* modeB, long* extentB, long* strideB, int typeBB,
            void* C_d, int nmodeC, int* modeC, long* extentC, long* strideC, int typeCC,
            void* D_d, int nmodeD, int* modeD, long* extentD, long* strideD, int typeDD,
            void* alpha, void* beta, int typeCompute2);

        public static void Einsum(
            void* A_d, int nmodeA, int* modeA, long* extentA, long* strideA, CudaDataType typeA,
            void* B_d, int nmodeB, int* modeB, long* extentB, long* strideB, CudaDataType typeB,
            void* C_d, int nmodeC, int* modeC, long* extentC, long* strideC, CudaDataType typeC,
            void* D_d, int nmodeD, int* modeD, long* extentD, long* strideD, CudaDataType typeD,
            void* alpha, void* beta, CutensorComputeType typeCompute)
        {
            int res = Einsum_Native(A_d, nmodeA, modeA, extentA, strideA, (int)typeA,
                B_d, nmodeB, modeB, extentB, strideB, (int)typeB,
                C_d, nmodeC, modeC, extentC, strideC, (int)typeC,
                D_d, nmodeD, modeD, extentD, strideD, (int)typeD, alpha, beta, (int)typeCompute);
            if (res != 0)
            {
                throw new Exception("Cutensor Einsum Error!");
            }
        }


        public static void Einsum(
            void* A_d, int nmodeA, int* modeA, long* extentA, long* strideA, CudaDataType typeA,
            void* B_d, int nmodeB, int* modeB, long* extentB, long* strideB, CudaDataType typeB,
            void* C_d, int nmodeC, int* modeC, long* extentC, long* strideC, CudaDataType typeC,
            void* D_d, int nmodeD, int* modeD, long* extentD, long* strideD, CudaDataType typeD,
            double alphaval, double betaval, CutensorComputeType typeCompute)
        {
            if (typeCompute == CutensorComputeType.CUTENSOR_COMPUTE_16F)
            {
                Half alpha = (Half)alphaval;
                Half beta = (Half)betaval;
                Einsum(A_d, nmodeA, modeA, extentA, strideA, typeA,
                           B_d, nmodeB, modeB, extentB, strideB, typeB,
                           C_d, nmodeC, modeC, extentC, strideC, typeC,
                           D_d, nmodeD, modeD, extentD, strideD, typeD, &alpha, &beta, typeCompute);
            }
            else if (typeCompute == CutensorComputeType.CUTENSOR_COMPUTE_32F)
            {
                float alpha = (float)alphaval;
                float beta = (float)betaval;
                Einsum(A_d, nmodeA, modeA, extentA, strideA, typeA,
                           B_d, nmodeB, modeB, extentB, strideB, typeB,
                           C_d, nmodeC, modeC, extentC, strideC, typeC,
                           D_d, nmodeD, modeD, extentD, strideD, typeD, &alpha, &beta, typeCompute);
            }
            else if (typeCompute == CutensorComputeType.CUTENSOR_COMPUTE_64F)
            {
                double alpha = (double)alphaval;
                double beta = (double)betaval;
                Einsum(A_d, nmodeA, modeA, extentA, strideA, typeA,
                          B_d, nmodeB, modeB, extentB, strideB, typeB,
                          C_d, nmodeC, modeC, extentC, strideC, typeC,
                          D_d, nmodeD, modeD, extentD, strideD, typeD, &alpha, &beta, typeCompute);
            }
            else if (typeCompute == CutensorComputeType.CUTENSOR_COMPUTE_32I)
            {
                int alpha = (int)alphaval;
                int beta = (int)betaval;
                Einsum(A_d, nmodeA, modeA, extentA, strideA, typeA,
                           B_d, nmodeB, modeB, extentB, strideB, typeB,
                           C_d, nmodeC, modeC, extentC, strideC, typeC,
                           D_d, nmodeD, modeD, extentD, strideD, typeD, &alpha, &beta, typeCompute);
            }
        }
        #endregion
    }

    public static class CudaTypes
    {
        public static CutensorComputeType GetComputeType(Tensor d)
        {
            if (d.Config.NumType == OptimizedNumerics.NumberType.Float16)
            {
                return CutensorComputeType.CUTENSOR_COMPUTE_16F;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Float32)
            {
                return CutensorComputeType.CUTENSOR_COMPUTE_32F;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Float64)
            {
                return CutensorComputeType.CUTENSOR_COMPUTE_64F;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Int32)
            {
                return CutensorComputeType.CUTENSOR_COMPUTE_32I;
            }

            throw new Exception("CutensorComputeType Error!");
        }

        public static CudaDataType GetDataType(Tensor d)
        {
            if (d.Config.NumType == OptimizedNumerics.NumberType.Float16)
            {
                return CudaDataType.CUDA_R_16F;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Float32)
            {
                return CudaDataType.CUDA_R_32F;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Float64)
            {
                return CudaDataType.CUDA_R_64F;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Int16)
            {
                return CudaDataType.CUDA_R_16I;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Int32)
            {
                return CudaDataType.CUDA_R_32I;
            }
            else if (d.Config.NumType == OptimizedNumerics.NumberType.Int64)
            {
                return CudaDataType.CUDA_R_64I;
            }

            throw new Exception("CutensorComputeType Error!");
        }

        public enum CudaDataType
        {
            CUDA_R_16F = 2, /* real as a half */
            CUDA_C_16F = 6, /* complex as a pair of half numbers */
            CUDA_R_16BF = 14, /* real as a nv_bfloat16 */
            CUDA_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
            CUDA_R_32F = 0, /* real as a float */
            CUDA_C_32F = 4, /* complex as a pair of float numbers */
            CUDA_R_64F = 1, /* real as a double */
            CUDA_C_64F = 5, /* complex as a pair of double numbers */
            CUDA_R_4I = 16, /* real as a signed 4-bit int */
            CUDA_C_4I = 17, /* complex as a pair of signed 4-bit int numbers */
            CUDA_R_4U = 18, /* real as a unsigned 4-bit int */
            CUDA_C_4U = 19, /* complex as a pair of unsigned 4-bit int numbers */
            CUDA_R_8I = 3, /* real as a signed 8-bit int */
            CUDA_C_8I = 7, /* complex as a pair of signed 8-bit int numbers */
            CUDA_R_8U = 8, /* real as a unsigned 8-bit int */
            CUDA_C_8U = 9, /* complex as a pair of unsigned 8-bit int numbers */
            CUDA_R_16I = 20, /* real as a signed 16-bit int */
            CUDA_C_16I = 21, /* complex as a pair of signed 16-bit int numbers */
            CUDA_R_16U = 22, /* real as a unsigned 16-bit int */
            CUDA_C_16U = 23, /* complex as a pair of unsigned 16-bit int numbers */
            CUDA_R_32I = 10, /* real as a signed 32-bit int */
            CUDA_C_32I = 11, /* complex as a pair of signed 32-bit int numbers */
            CUDA_R_32U = 12, /* real as a unsigned 32-bit int */
            CUDA_C_32U = 13, /* complex as a pair of unsigned 32-bit int numbers */
            CUDA_R_64I = 24, /* real as a signed 64-bit int */
            CUDA_C_64I = 25, /* complex as a pair of signed 64-bit int numbers */
            CUDA_R_64U = 26, /* real as a unsigned 64-bit int */
            CUDA_C_64U = 27  /* complex as a pair of unsigned 64-bit int numbers */
        }

        public enum CutensorComputeType
        {
            CUTENSOR_COMPUTE_16F = (1 << 0),  ///< floating-point: 5-bit exponent and 10-bit mantissa (aka half)
            CUTENSOR_COMPUTE_16BF = (1 << 10),  ///< floating-point: 8-bit exponent and 7-bit mantissa (aka bfloat)
            CUTENSOR_COMPUTE_TF32 = (1 << 12),  ///< floating-point: 8-bit exponent and 10-bit mantissa (aka tensor-float-32)
            CUTENSOR_COMPUTE_32F = (1 << 2),  ///< floating-point: 8-bit exponent and 23-bit mantissa (aka float)
            CUTENSOR_COMPUTE_64F = (1 << 4),  ///< floating-point: 11-bit exponent and 52-bit mantissa (aka double)
            CUTENSOR_COMPUTE_8U = (1 << 6),  ///< 8-bit unsigned integer
            CUTENSOR_COMPUTE_8I = (1 << 8),  ///< 8-bit signed integer
            CUTENSOR_COMPUTE_32U = (1 << 7),  ///< 32-bit unsigned integer
            CUTENSOR_COMPUTE_32I = (1 << 9),  ///< 32-bit signed integer
        }

    }


}
