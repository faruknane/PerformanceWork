using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text;

namespace PerformanceWork
{
    public class ArrayPool<T> : IDisposable
    {
        public int TimesLarger
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set;
        }
        public int MaxCount
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set;
        }
        public int Count { [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get; private set;
        } = 0;

        public int UnreturnedArrayCount
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set;
        }

        FloatFastNode<T> first;
        FloatFastNode<T> last;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ArrayPool(int timeslarger, int maxarraycount)
        {
            TimesLarger = timeslarger;
            MaxCount = maxarraycount;
            first = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ArrayPool<T> Create(int timeslarger, int maxarraycount)
            => new ArrayPool<T>(timeslarger, maxarraycount);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T[] Rent(int minlength)
        {
            UnreturnedArrayCount++;
            T[] arr = FindandExtractArray(minlength);
            if(arr == null)
            {
                arr = new T[minlength];
                return arr;
            }
            return arr;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private T[] FindandExtractArray(int minlength)
        {
            FloatFastNode<T> select = null;
            int min = int.MaxValue;
            FloatFastNode<T> head = first;
            while(head != null)
            {
                int mylength = head.Array.Length;
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
                T[] res = select.Array;

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool Condition(int minlength, int mylength)
        {
            if (mylength >= minlength && mylength <= minlength * TimesLarger) return true;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Return(T[] array)
        {
            if (Count == MaxCount)
                throw new Exception("ArrayPool Maxcount has been reached!");

            var n = new FloatFastNode<T>(array);
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ClearMemory()
        {
            first = last = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Dispose()
        {
            ClearMemory();
        }
    }
}
