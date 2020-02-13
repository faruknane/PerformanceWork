using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Shape : IDisposable
    {
        public static ArrayPool<int> ShapePool = ArrayPool<int>.Create(10, 10);
        public static ObjectPool<Shape> ObjectPool = new ObjectPool<Shape>();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape NewShapeN(int n)
        {
            Shape s = ObjectPool.Rent();
            if (s == null)
                s = new Shape(n);
            else
            {
                s.ArrayReturned = false;
                s.N = n;
                s.Dimensions = (int*)ShapePool.Rent(s.N, out s.Length1);
                s.Multiplied = (int*)ShapePool.Rent(s.N + 1, out s.Length2);
            }
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape NewShape(int x0)
        {
            Shape s = Shape.NewShapeN(1);
            s.Dimensions[0] = x0;
            s.CalculateMultiplied();
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape NewShape(int x0, int x1)
        {
            Shape s = Shape.NewShapeN(2);
            s.Dimensions[0] = x0;
            s.Dimensions[1] = x1;
            s.CalculateMultiplied();
            return s;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape NewShape(int x0, int x1, int x2)
        {
            Shape s = Shape.NewShapeN(3);
            s.Dimensions[0] = x0;
            s.Dimensions[1] = x1;
            s.Dimensions[2] = x2;
            s.CalculateMultiplied();
            return s;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape NewShape(int x0, int x1, int x2, int x3)
        {
            Shape s = Shape.NewShapeN(4);
            s.Dimensions[0] = x0;
            s.Dimensions[1] = x1;
            s.Dimensions[2] = x2;
            s.Dimensions[3] = x3;
            s.CalculateMultiplied();
            return s;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape NewShape(int x0, int x1, int x2, int x3, int x4)
        {
            Shape s = Shape.NewShapeN(5);
            s.Dimensions[0] = x0;
            s.Dimensions[1] = x1;
            s.Dimensions[2] = x2;
            s.Dimensions[3] = x3;
            s.Dimensions[4] = x4;
            s.CalculateMultiplied();
            return s;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Return(Shape s)
        {
            if (!s.ArrayReturned)
                s.Dispose();
            ObjectPool.Return(s);
        }


        public int* Dimensions;
        public int* Multiplied;

        public int N { get; private set; }
        public int TotalSize { get => Multiplied[0]; }

        private int Length1;
        private int Length2;
        private bool ArrayReturned = false;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape(int n)
        {
            this.N = n;
            Dimensions = (int*)ShapePool.Rent(N, out Length1);
            Multiplied = (int*)ShapePool.Rent(N+1, out Length2);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape(params int[] dims)
        {
            if (dims.Length == 0)
                throw new Exception("The array has no element!");

            N = dims.Length;

            Dimensions = (int*)ShapePool.Rent(N, out Length1);
            Multiplied = (int*)ShapePool.Rent(N+1, out Length2);
            Multiplied[N] = 1;
            for (int i = N - 1; i >= 0; i--)
            {
                Dimensions[i] = dims[i];
                Multiplied[i] = Dimensions[i] * Multiplied[i + 1];
            }
        }

        public int this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get => Dimensions[x];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void CalculateMultiplied()
        {
            Multiplied[N] = 1;
            for (int i = N - 1; i >= 0; i--)
                Multiplied[i] = Dimensions[i] * Multiplied[i + 1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int[] dims)
        {
            int res = 0;
            for (int i = 0; i < dims.Length; i++)
                res += dims[i] * Multiplied[i + 1];
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int x1)
        {
            return x1 * Multiplied[1];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int x1, int x2)
        {
            return x1 * Multiplied[1] + x2 * Multiplied[2];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int x1, int x2, int x3)
        {
            return x1 * Multiplied[1] + x2 * Multiplied[2] + x3 * Multiplied[3];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int x1, int x2, int x3, int x4)
        {
            return x1 * Multiplied[1] + x2 * Multiplied[2] + x3 * Multiplied[3] + x4 * Multiplied[4];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int x1, int x2, int x3, int x4, int x5)
        {
            return x1 * Multiplied[1] + x2 * Multiplied[2] + x3 * Multiplied[3] + x4 * Multiplied[4] + x5 * Multiplied[5];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(int x1, int x2, int x3, int x4, int x5, int x6)
        {
            return x1 * Multiplied[1] + x2 * Multiplied[2] + x3 * Multiplied[3] + x4 * Multiplied[4] + x5 * Multiplied[5] + x6 * Multiplied[6];
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape Clone()
        {
            Shape s = Shape.NewShapeN(this.N);
            for (int i = 0; i < s.N; i++)
            {
                s.Dimensions[i] = this.Dimensions[i];
                s.Multiplied[i] = this.Multiplied[i];
            }
            s.Multiplied[this.N] = 1;
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool EqualShape(Shape s)
        {
            if (this.N != s.N) return false;
            for (int i = 0; i < N; i++)
                if (this[i] != s[i])
                    return false;
            return true;
        }

        public override string ToString()
        {
            string res = "";
            for (int i = 0; i < N; i++)
                res += Dimensions[i] + ", ";
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            if (ArrayReturned)
                throw new Exception("The shape object is already returned!");
            ArrayReturned = true;

            ShapePool.Return(Dimensions, Length1);
            ShapePool.Return(Multiplied, Length2);
            GC.SuppressFinalize(this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape Combine(Shape s1, Shape s2)
        {
            Shape c = Shape.NewShapeN(s1.N + s2.N);
            for (int i = 0; i < s1.N + s2.N; i++)
            {
                if (i < s1.N)
                    c.Dimensions[i] = s1.Dimensions[i];
                else
                    c.Dimensions[i] = s2.Dimensions[i - s1.N];
            }
            c.CalculateMultiplied();
            return c;
        }


        /// <summary>
        /// returns s1 - s2 + s3;
        /// </summary>
        /// <param name="s1"></param>
        /// <param name="s2"></param>
        /// <param name="s3"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape SwapTail(Shape s1, Shape s2, Shape s3)
        {
            Shape c = Shape.NewShapeN(s1.N - s2.N + s3.N);

            for (int i = 0; i < s1.N - s2.N; i++)
                c.Dimensions[i] = s1.Dimensions[i];

            for (int i = s1.N - s2.N; i < s1.N - s2.N + s3.N; i++)
                c.Dimensions[i] = s3.Dimensions[i - (s1.N - s2.N)];

            c.CalculateMultiplied();
            return c;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape Divide(Shape s1, Shape s2)
        {
            if (s1.N != s2.N)
                throw new Exception("dimensions incompatibility!");

            Shape res = Shape.NewShapeN(s1.N);
            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s1[i] / s2[i];
            res.CalculateMultiplied();

            return res;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape Multiply(Shape s1, Shape s2)
        {
            if (s1.N != s2.N)
                throw new Exception("dimensions incompatibility!");

            Shape res = Shape.NewShapeN(s1.N);
            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s1[i] * s2[i];
            res.CalculateMultiplied();

            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape RemoveLastDimension(Shape s)
        {
            Shape res = Shape.NewShapeN(s.N - 1);

            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s.Dimensions[i];

            res.CalculateMultiplied();

            return res;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        ~Shape()
        {
            if (!this.ArrayReturned)
            {
                this.Dispose();
            }
        }
    }
}
