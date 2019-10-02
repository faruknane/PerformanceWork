using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork
{
    public unsafe class ArrayPool<T> : IDisposable
    {
        public int TimesLarger
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }
        public int MaxCount
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }
        public int Count { [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get; private set;
        } = 0;

        public int UnreturnedArrayCount
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        FloatFastNode first;
        FloatFastNode last;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private ArrayPool(int timeslarger, int maxarraycount)
        {
            TimesLarger = timeslarger;
            MaxCount = maxarraycount;
            first = null;
            UnreturnedArrayCount = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool<T> Create(int timeslarger, int maxarraycount)
            => new ArrayPool<T>(timeslarger, maxarraycount);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* Rent(int minlength, out int l)
        {
            //System.Diagnostics.StackTrace a = new System.Diagnostics.StackTrace();
            //foreach (var item in a.GetFrames())
            //{
            //    Console.Write(item.ToString());
            //    break;
            //}
            UnreturnedArrayCount++;
            void* arr = FindandExtractArray(minlength, out l);
            if(arr == null)
            {
                arr = MKL.MKL_malloc(minlength * Marshal.SizeOf<T>(), 32);
                l = minlength;
                return arr;
            }
            return arr;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void* FindandExtractArray(int minlength, out int l)
        {
            l = -1;
            FloatFastNode select = null;
            int min = int.MaxValue;
            FloatFastNode head = first;
            while(head != null)
            {
                int mylength = head.Length;
                if (Condition(minlength, mylength))
                {
                    if (min > mylength)
                    {
                        min = mylength;
                        select = head;
                    }
                }
                head = head.R;
            }

            if (select == null)
                return null;
            else
            {
                Count--;
                void* res = select.Array;
                l = select.Length;
                if (select.L != null)
                    select.L.R = select.R;
                else 
                    first = select.R;

                if (select.R != null)
                    select.R.L = select.L;
                else
                    last = select.L;
                return res;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private bool Condition(int minlength, int mylength)
        {
            if (mylength >= minlength && mylength <= minlength * TimesLarger) return true;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(void* array, int l)
        {
            if (Count == MaxCount)
                throw new Exception("ArrayPool Maxcount has been reached!");

            var n = new FloatFastNode(array, l);
            Count++;
            if (last == null)
            {
                first = last = n;
            }
            else
            {
                last.R = n;
                n.L = last;
                last = n;
            }
            UnreturnedArrayCount--;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void ClearMemory()
        {
            first = last = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            ClearMemory();
        }
    }
}
