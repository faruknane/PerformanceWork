using PerformanceWork.NCuda;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics.Pool
{
    /// <summary>
    /// Thread Safe Array Pool
    /// </summary>
    public unsafe class ArrayPool : IDisposable
    {
        public unsafe struct PointerArray
        {
            public void* Ptr;
        }

        public int UnreturnedArrayCount { get; private set; } = 0;

        private Dictionary<long, Stack<PointerArray>> Stacks;

        public Device Device;

        private readonly object Mutex = new object();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ArrayPool(Device dev)
        {
            Stacks = new Dictionary<long, Stack<PointerArray>>();
            Device = dev;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* Rent(long length, long unitlength)
        {
            lock (Mutex)
            {
                UnreturnedArrayCount++;

                length *= unitlength;
                if (Stacks.ContainsKey(length) && (Stacks[length] is Stack<PointerArray> x) && x.Count > 0)
                {
                    PointerArray sr = x.Pop();
                    return sr.Ptr;
                }
                else
                {
                    if (this.Device.Type == DeviceType.NvidiaGPU)
                    {
                        GC.AddMemoryPressure(length);
                        CudaManagement.SetDevice(this.Device.ID);
                        return CudaManagement.Allocate(length, Device.ID);
                    }
                    else if (this.Device.Type == DeviceType.Host)
                    {
                        GC.AddMemoryPressure(length);
                        return MKL.MKL_malloc(length, 32);
                    }
                    else
                        throw new Exception("Uknown Device in ArrayPool!");
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(void* arr, long length, long unitlength)
        {
            lock (Mutex)
            {
                length *= unitlength;
                PointerArray sr = new PointerArray();
                sr.Ptr = arr;

                (Stacks.ContainsKey(length) ? Stacks[length] : (Stacks[length] = new Stack<PointerArray>())).Push(sr);

                UnreturnedArrayCount--;
            }
        }

        public void EraseAll()
        {
            lock (Mutex)
            {
                foreach (var item in Stacks)
                {
                    if (item.Value != null)
                    {
                        foreach (var arr in item.Value)
                        {
                            if (this.Device.Type == DeviceType.NvidiaGPU)
                            {
                                CudaManagement.SetDevice(this.Device.ID);
                                CudaManagement.Free(arr.Ptr);
                            }
                            else if (this.Device.Type == DeviceType.Host)
                            {
                                GC.RemoveMemoryPressure((long)item.Key);
                                MKL.MKL_free(arr.Ptr);
                            }
                            else
                                throw new Exception("Uknown Device in ArrayPool!");
                        }
                        item.Value.Clear();
                    }
                }
                Stacks.Clear();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            EraseAll();
        }

    }
}
