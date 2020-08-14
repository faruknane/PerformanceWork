using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Shape
    {
        public int[] Dimensions;
        public int[] Multiplied;

        public int N { get; private set; }
        public int TotalSize { get => Multiplied[0]; }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape(int n)
        {
            this.N = n;
            Dimensions = new int[N];
            Multiplied = new int[N + 1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape((int x1, bool rastgele) a)
        {
            this.N = 1;
            Dimensions = new int[N];
            Multiplied = new int[N + 1];
            Multiplied[N] = 1;

            Dimensions[0] = a.x1;
            Multiplied[0] = a.x1 * Multiplied[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape((int x1, int x2) a)
        {
            this.N = 2;
            Dimensions = new int[N];
            Multiplied = new int[N + 1];
            Multiplied[N] = 1;

            Dimensions[1] = a.x2;
            Multiplied[1] = a.x2 * Multiplied[2];

            Dimensions[0] = a.x1;
            Multiplied[0] = a.x1 * Multiplied[1];

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape((int x1, int x2, int x3) a)
        {
            this.N = 3;
            Dimensions = new int[N];
            Multiplied = new int[N + 1];
            Multiplied[N] = 1;

            Dimensions[2] = a.x3;
            Multiplied[2] = a.x3 * Multiplied[3];

            Dimensions[1] = a.x2;
            Multiplied[1] = a.x2 * Multiplied[2];

            Dimensions[0] = a.x1;
            Multiplied[0] = a.x1 * Multiplied[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape((int x1, int x2, int x3, int x4) a)
        {
            this.N = 4;
            Dimensions = new int[N];
            Multiplied = new int[N + 1];
            Multiplied[N] = 1;

            Dimensions[3] = a.x4;
            Multiplied[3] = a.x4 * Multiplied[4];

            Dimensions[2] = a.x3;
            Multiplied[2] = a.x3 * Multiplied[3];

            Dimensions[1] = a.x2;
            Multiplied[1] = a.x2 * Multiplied[2];

            Dimensions[0] = a.x1;
            Multiplied[0] = a.x1 * Multiplied[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape((int x1, int x2, int x3, int x4, int x5) a)
        {
            this.N = 5;
            Dimensions = new int[N];
            Multiplied = new int[N + 1];
            Multiplied[N] = 1;

            Dimensions[4] = a.x5;
            Multiplied[4] = a.x5 * Multiplied[5];

            Dimensions[3] = a.x4;
            Multiplied[3] = a.x4 * Multiplied[4];

            Dimensions[2] = a.x3;
            Multiplied[2] = a.x3 * Multiplied[3];

            Dimensions[1] = a.x2;
            Multiplied[1] = a.x2 * Multiplied[2];

            Dimensions[0] = a.x1;
            Multiplied[0] = a.x1 * Multiplied[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape(params int[] dims)
        {
            if (dims.Length == 0)
                throw new Exception("The array has no element!");

            N = dims.Length;

            Dimensions = new int[N];
            Multiplied = new int[N + 1];
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

        //todo create unit test for index methods
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int Index(Index ind)
        {
            int res = 0;
            for (int i = 0; i < ind.N; i++)
                res += ind[i] * Multiplied[i + 1];
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
            Shape s = new Shape(this.N);
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
        public static Shape Combine(Shape s1, Shape s2)
        {
            Shape c = new Shape(s1.N + s2.N);
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
            Shape c = new Shape(s1.N - s2.N + s3.N);

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

            Shape res = new Shape(s1.N);
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

            Shape res = new Shape(s1.N);
            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s1[i] * s2[i];
            res.CalculateMultiplied();

            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape RemoveLastDimension(Shape s)
        {
            Shape res = new Shape(s.N - 1);

            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s.Dimensions[i];

            res.CalculateMultiplied();

            return res;
        }


    
    }
}
