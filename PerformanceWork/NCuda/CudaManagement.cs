using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;

namespace PerformanceWork.NCuda
{
    public static unsafe class CudaManagement
    {
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NFree"), SuppressUnmanagedCodeSecurity]
        public static extern void Free(void* arr);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NAllocate"), SuppressUnmanagedCodeSecurity]
        public static extern void* Allocate(long bytesize, int gpuid);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NGetDevice"), SuppressUnmanagedCodeSecurity]
        public static extern int GetDevice();

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NSetDevice"), SuppressUnmanagedCodeSecurity]
        public static extern void SetDevice(int gpuid);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NCopyArray"), SuppressUnmanagedCodeSecurity]
        public static extern void CopyArray(void* src, void* dst, long bytesize);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NCheckError"), SuppressUnmanagedCodeSecurity]
        public static extern void CheckError();

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false, EntryPoint = "NDeviceSynchronize"), SuppressUnmanagedCodeSecurity]
        public static extern void DeviceSynchronize();

    }
}
