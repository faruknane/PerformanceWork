using System;
using System.Collections.Generic;
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

        public bool HasRight { get => R != null; }
        public bool HasLeft { get => L != null; }
        public void SetLeft(ref FloatFastNode<T> n) => L = n;
        public void SetRight(ref FloatFastNode<T> n) => R = n;
        public FloatFastNode(T[] arr) => Array = arr;
    }


}
