using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;

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

    }
}
