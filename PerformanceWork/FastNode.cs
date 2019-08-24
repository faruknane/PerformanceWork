using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork
{
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public unsafe class FloatFastNode<T>
    {
        public FloatFastNode<T> L;
        public FloatFastNode<T> R;
        public T[] Array;

        public bool HasRight
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => R != null;
        }
        public bool HasLeft
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => L != null;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetLeft(ref FloatFastNode<T> n) => L = n;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetRight(ref FloatFastNode<T> n) => R = n;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FloatFastNode(T[] arr) => Array = arr;
    }


}