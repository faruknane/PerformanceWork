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
        #region No Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat32(float* res, float* a, float* b, long length);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat64(float* res, float* a, float* b, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt16(short* res, short* a, short* b, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt32(int* res, int* a, int* b, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt64(long* res, long* a, long* b, long length);
        #endregion
        #region With Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddFloat32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat32(float* res, float* a, float* b, long length, float cofa, float cofb, float cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddFloat64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat64(double* res, double* a, double* b, long length, double cofa, double cofb, double cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddInt16_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt16(short* res, short* a, short* b, long length, short cofa, short cofb, short cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddInt32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt32(int* res, int* a, int* b, long length, int cofa, int cofb, int cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddInt64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt64(long* res, long* a, long* b, long length, long cofa, long cofb, long cofadd);
        #endregion
        #endregion

        #region Assign Vector Operation
        #region No Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat32(float* res, float* a, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat64(double* res, double* a, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt16(short* res, short* a, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt32(int* res, int* a, long length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt64(long* res, long* a, long length);
        #endregion
        #region With Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignFloat32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat32(float* res, float* a, long length, float alpha, float beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignFloat64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat64(double* res, double* a, long length, double alpha, double beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignInt16_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt16(short* res, short* a, long length, short alpha, short beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignInt32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt32(int* res, int* a, long length, int alpha, int beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignInt64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt64(long* res, long* a, long length, long alpha, long beta);

        #endregion
        #endregion



    }
}
