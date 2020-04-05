using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics.Pool
{
    public unsafe class ArrayPool : IDisposable
    {
        public int MaxLength { get; private set; }
        public int BucketCount { get; private set; }
        public int BucketSize { get; private set; }
        public int UnreturnedArrayCount { get; private set; } = 0;      

        public Stack<PointerArray>[] Stacks;

        public int DeviceId { get; private set; }
        public bool OnGPU { get; private set; }

        private object Mutex = new object();
        //todo deviceindicator
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ArrayPool(int MaxLength, int BucketCount,  bool gpu = false, int devid = -1)
        {
            this.MaxLength = MaxLength;
            this.BucketCount = BucketCount;
            this.BucketSize = (MaxLength + BucketCount - 1) / BucketCount;
            Stacks = new Stack<PointerArray>[BucketCount + 1];
            OnGPU = gpu;
            DeviceId = devid;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* Rent(int minlength, out int length, Data.Type t)
        {
            lock (Mutex)
            {
                UnreturnedArrayCount++;
                // 0 -> 0, BucketSize - 1
                // 1 -> BucketSize, 2*BucketSize-1
                int bucketno = minlength / BucketSize + 1;
                if (Stacks[bucketno] != null && Stacks[bucketno].Count > 0)
                {
                    PointerArray sr = Stacks[bucketno].Pop();
                    length = sr.Length;
                    return sr.Ptr;
                }
                else
                {
                    length = minlength + BucketSize;
                    if (OnGPU)
                        return NCuda.Allocate(length * Data.GetByteSize(t), DeviceId);
                    else
                        return MKL.MKL_malloc(length * Data.GetByteSize(t), 32);
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
                sr.Length = length;
                int bucketno = length / BucketSize;

                if (Stacks[bucketno] == null)
                    Stacks[bucketno] = new Stack<PointerArray>();
                Stacks[bucketno].Push(sr);
                UnreturnedArrayCount--;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool Create(int MaxLength, int BucketCount)
        { 
            return new ArrayPool(MaxLength, BucketCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool Create(int MaxLength, int BucketCount, bool OnGPU, int DeviceId)
        {
            return new ArrayPool(MaxLength, BucketCount, OnGPU, DeviceId);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            throw new NotImplementedException();//Deallocation
        }
    }
}
