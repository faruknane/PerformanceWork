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
        public static extern void AddFloat32(float* res, float* a, float* b, int length);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat64(float* res, float* a, float* b, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt16(short* res, short* a, short* b, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt32(int* res, int* a, int* b, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt64(long* res, long* a, long* b, long length);
        #endregion
        #region With Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddFloat32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat32(float* res, float* a, float* b, int length, float cofa, float cofb, float cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddFloat64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddFloat64(double* res, double* a, double* b, int length, double cofa, double cofb, double cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddInt16_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt16(short* res, short* a, short* b, int length, short cofa, short cofb, short cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddInt32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt32(int* res, int* a, int* b, int length, int cofa, int cofb, int cofadd);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AddInt64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AddInt64(long* res, long* a, long* b, int length, long cofa, long cofb, long cofadd);
        #endregion
        #endregion

        #region Assign Vector Operation
        #region No Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat32(float* res, float* a, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat64(double* res, double* a, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt16(short* res, short* a, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt32(int* res, int* a, int length);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt64(long* res, long* a, int length);
        #endregion
        #region With Coefficient
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignFloat32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat32(float* res, float* a, int length, float alpha, float beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignFloat64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignFloat64(double* res, double* a, int length, double alpha, double beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignInt16_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt16(short* res, short* a, int length, short alpha, short beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignInt32_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt32(int* res, int* a, int length, int alpha, int beta);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "AssignInt64_Coefficients"), SuppressUnmanagedCodeSecurity]
        public static extern void AssignInt64(long* res, long* a, int length, long alpha, long beta);

        #endregion
        #endregion



    }
}
