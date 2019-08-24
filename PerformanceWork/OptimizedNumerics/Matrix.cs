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
        public Matrix(float[,] arr)
        {
            D1 = arr.GetLength(0);
            D2 = arr.GetLength(1);
            Array = pool.Rent(D1 * D2);
            for (int i = 0; i < D1; i++)
                for (int j = 0; j < D2; j++)
                    this[i, j] = arr[i, j];
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

        public float this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Array[x];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Array[x ] = value;
            }
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override bool Equals(object obj)
        {
            if (base.Equals(obj)) return true;
            if (!(obj is Matrix)) return false;

            Matrix o = (Matrix)obj;

            if (this.D1 != o.D1 || this.D2 != o.D2) return false;

            return Vectorization.EqualsAVX(this.Array, o.Array, this.Array.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.D1 != b.D1 || a.D2 != b.D2)
                throw new Exception("Matrices to be sum should have the same dimensions!");

            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.AddAVX(a.Array, b.Array, res.Array, res.Array.Length);
            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix operator +(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.AddAVX(a.Array, b, res.Array, res.Array.Length);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.D1 != b.D1 || a.D2 != b.D2)
                throw new Exception("Matrices to be sum should have the same dimensions!");

            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.SubtractAVX(a.Array, b.Array, res.Array, res.Array.Length);
            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix operator -(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.AddAVX(a.Array, -b, res.Array, res.Array.Length);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix operator *(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.MultiplyAVX(a.Array, b, res.Array, res.Array.Length);
            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix operator /(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.MultiplyAVX(a.Array, 1 / b, res.Array, res.Array.Length);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Matrix(float[,] a)
        {
            return new Matrix(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Matrix(float a)
        {
            Matrix m = new Matrix(1, 1);
            m[0] = a;
            return m;
        }
    }
}
