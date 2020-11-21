using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics.Pool
{
    public unsafe class ArrayPool : IDisposable
    {
        public unsafe struct PointerArray
        {
            public void* Ptr;
        }

        public int MaxLength { get; private set; }
        public int UnreturnedArrayCount { get; private set; } = 0;      

        public Stack<PointerArray>[] Stacks;

        public DeviceConfig DevConfig;

        private readonly object Mutex = new object();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ArrayPool(int MaxLength, DeviceConfig devconf)
        {
            this.MaxLength = MaxLength;
            Stacks = new Stack<PointerArray>[MaxLength];
            DevConfig = devconf;
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //public void* Rent(int length)
        //{
        //    return Rent(length, DevConfig.GetByteLength());
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* Rent(int length, int unitlength)
        {
            lock (Mutex)
            {
                UnreturnedArrayCount++;

                if (Stacks[length] != null && Stacks[length].Count > 0)
                {
                    PointerArray sr = Stacks[length].Pop();
                    return sr.Ptr;
                }
                else
                {
                    if (this.DevConfig.DevType == DeviceConfig.DeviceType.NvidiaGPU)
                    {
                        GC.AddMemoryPressure(length * unitlength);
                        return NCuda.Allocate(length * unitlength, DevConfig.DeviceID);
                    }
                    else if (this.DevConfig.DevType == DeviceConfig.DeviceType.Host)
                    {
                        GC.AddMemoryPressure(length * unitlength);
                        return MKL.MKL_malloc(length * unitlength, 32);
                    }
                    else
                        throw new Exception("Uknown Device in ArrayPool!");
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(void* arr, int length)
        {
            lock (Mutex)
            {
                PointerArray sr = new PointerArray();
                sr.Ptr = arr;

                (Stacks[length] ?? (Stacks[length] = new Stack<PointerArray>())).Push(sr);

                UnreturnedArrayCount--;
            }
        }

        public void EraseAll()
        {
            lock (Mutex)
            {
                if (this.DevConfig.DevType == DeviceConfig.DeviceType.NvidiaGPU)
                {
                    for (int i = 0; i < Stacks.Length; i++)
                    {
                        Stack<PointerArray> s = Stacks[i];
                        if (s != null)
                        {
                            foreach (var arr in s)
                            {
                                NCuda.Free(arr.Ptr, DevConfig.DeviceID);
                            }
                            s.Clear();
                        }
                    }
                }
                else if (this.DevConfig.DevType == DeviceConfig.DeviceType.Host)
                {
                    for (int i = 0; i < Stacks.Length; i++)
                    {
                        Stack<PointerArray> s = Stacks[i];
                        if (s != null)
                        {
                            foreach (var arr in s)
                            {
                                MKL.MKL_free(arr.Ptr);
                            }
                            s.Clear();
                        }
                    }
                }
                else
                    throw new Exception("Uknown Device in ArrayPool!");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            EraseAll();
        }

    }
}
