using System;
using System.Runtime.CompilerServices;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe struct Matrix : IDisposable
    {
        public float[] Array;
        public int D1, D2;
        internal static ArrayPool<float> pool = ArrayPool<float>.Create(2, 50);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix(int d1, int d2)
        {
            D1 = d1;
            D2 = d2;
            Array = pool.Rent(D1 * D2);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Dispose()
        {
            pool.Return(Array);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal float* GetPointer()
        {
            fixed (float* ptr = Array)
                return ptr;
        }
        public float this[int x, int y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Array[x * D2 + y];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Array[x * D2 + y] = value;
            }
        }

        public float this[long x, long y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Array[x * D2 + y];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Array[x * D2 + y] = value;
            }
        }

        public Matrix TransposeMatrix()
        {
            return this;
        }
    }
}
