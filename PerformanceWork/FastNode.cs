using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork
{
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public unsafe class FloatFastNode
    {
        public FloatFastNode L;
        public FloatFastNode R;
        public void* Array;
        public int Length;
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
        public void SetLeft(ref FloatFastNode n) => L = n;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetRight(ref FloatFastNode n) => R = n;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe FloatFastNode(void* arr, int length)
        {
            this.Length = length;
            Array = arr;
        }
    }


}