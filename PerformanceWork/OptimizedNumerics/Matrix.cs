using System;
using System.Runtime.CompilerServices;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Matrix : IDisposable
    {   
        public float* Array;
        private int Length;

        public int D1, D2;
        public static ArrayPool<float> Pool = ArrayPool<float>.Create(2, 1350);
        public static object l = new object();
        public bool Returned { get; private set; } = false;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Matrix(int d1, int d2)
        {
            D1 = d1;
            D2 = d2;
            lock(l)
                Array = (float*)Pool.Rent(D1 * D2, out Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Matrix(float[,] arr)
        {
            D1 = arr.GetLength(0);
            D2 = arr.GetLength(1);
            lock (l)
                Array = (float*)Pool.Rent(D1 * D2, out Length);
            for (int i = 0; i < D1; i++)
                for (int j = 0; j < D2; j++)
                    this[i, j] = arr[i, j];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetZero()
        {
            Vectorization.ElementWiseSetValueAVX(Array, 0, D1*D2);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            GC.SuppressFinalize(this);

            if (Returned)
                throw new Exception("Already Returned!");
            Returned = true;
            lock (l)
                Pool.Return(Array, Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float* GetPointer()
        {
            return Array;
        }

        public float this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                return Array[x];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                Array[x ] = value;
            }
        }

        public float this[int x, int y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                return Array[x * D2 + y];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                Array[x * D2 + y] = value;
            }
        }

        public float this[long x, long y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                return Array[x * D2 + y];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                Array[x * D2 + y] = value;
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override bool Equals(object obj)
        {
            if (base.Equals(obj)) return true;
            if (!(obj is Matrix)) return false;

            Matrix o = (Matrix)obj;

            if (this.D1 != o.D1 || this.D2 != o.D2) return false;

            return Vectorization.ElementWiseIsEqualsAVX(this.Array, o.Array, D1*D2);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.D1 != b.D1 || a.D2 != b.D2)
                throw new Exception("Matrices to be sum should have same dimensions!");

            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.ElementWiseAddAVX(a.Array, b.Array, res.Array, a.D1 * a.D2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix operator +(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.ElementWiseAddAVX(a.Array, b, res.Array, a.D1 * a.D2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.D1 != b.D1 || a.D2 != b.D2)
                throw new Exception("Matrices to be sum should have same dimensions!");

            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.ElementWiseSubtractAVX(a.Array, b.Array, res.Array, a.D1 * a.D2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix operator -(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.ElementWiseAddAVX(a.Array, -b, res.Array, a.D1 * a.D2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix operator *(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.ElementWiseMultiplyAVX(a.Array, b, res.Array, a.D1 * a.D2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix operator /(Matrix a, float b)
        {
            Matrix res = new Matrix(a.D1, a.D2);
            Vectorization.ElementWiseMultiplyAVX(a.Array, 1 / b, res.Array, a.D1 * a.D2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator Matrix(float[,] a)
        {
            return new Matrix(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator Matrix(float a)
        {
            Matrix m = new Matrix(1, 1);
            m[0] = a;
            return m;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void ElementWiseMultiply(Matrix b)
        {
            if (this.D1 != b.D1 || this.D2 != b.D2)
                throw new Exception("The dimensions should be same!");

            Vectorization.ElementWiseMultiplyAVX(this.Array, b.Array, this.Array, D1 * D2);
        }

        #region Static Methods
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix ElementWiseMultiply(Matrix a, Matrix b)
        {
            if (a.D1 != b.D1 || a.D2 != b.D2)
                throw new Exception("The dimensions should be same!");

            Matrix c = new Matrix(a.D1,a.D2);
            Vectorization.ElementWiseMultiplyAVX(a.Array,b.Array,c.Array, a.D1 * a.D2);
            return c;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix CreateCopy(Matrix m)
        {
            Matrix c = new Matrix(m.D1,m.D2);
            Vectorization.ElementWiseAssignAVX(c.Array, m.Array, m.D1 * m.D2);
            return c;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix MatrixMultiply(Matrix a, Matrix b)
        {
            Matrix c = new Matrix(a.D1, b.D2);
            Vectorization.MatrixMultiply(a, b, c);
            return c;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Matrix TranposeOf(Matrix x)
        {
            Matrix m = new Matrix(x.D2, x.D1);
            for (int i = 0; i < x.D1; i++)
                for (int j = 0; j < x.D2; j++)
                        m[j, i] = x[i, j];
            return m;
        }
        #endregion
    }
}
