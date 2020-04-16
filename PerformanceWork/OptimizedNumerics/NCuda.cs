using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public static unsafe class NCuda
    {
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void NFree(void* arr);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void* NAllocate(int bytesize, int gpuid);

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern int NGetDevice();

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void NSetDevice(int gpuid);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void NCopyFromHostToGPU(void* src, void* dst, int bytesize);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void NCopyFromGPUToHost(void* src, void* dst, int bytesize);
        
        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void NCheckError();

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void NDeviceSynchronize();

        [DllImport("NCuda\\NCuda.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false), SuppressUnmanagedCodeSecurity]
        public static extern void addWithCuda(float* a, float* b, float* res, int size);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Free(void* arr, int deviceId)
        {
            NSetDevice(deviceId);
            NFree(arr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void* Allocate(int bytesize, int deviceId)
        {
            return NAllocate(bytesize, deviceId);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int GetDevice()
        {
            return NGetDevice();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SetDevice(int gpuid)
        {
            NSetDevice(gpuid);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void CopyFromHostToGPU(void* src, void* dst, int bytesize)
        {
            NCopyFromHostToGPU(src, dst, bytesize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void CopyFromGPUToHost(void* src, void* dst, int bytesize)
        {
            NCopyFromGPUToHost(src, dst, bytesize);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void CheckError()
        {
            NCheckError();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void DeviceSynchronize()
        {
            NDeviceSynchronize();
        }
    }
}
