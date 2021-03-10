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
        public long[] Dimensions;
        public long[] Multiplied;

        public int N { get; private set; }
        public long TotalSize { get => Multiplied[0]; }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Shape()
        {

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape DimensionOf(int n)
        {
            Shape s = new Shape();
            s.N = n;
            s.Dimensions = new long[n];
            s.Multiplied = new long[n + 1];
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape(params long[] dims)
        {
            if (dims.Length == 0)
                throw new Exception("The array has no element!");

            N = dims.Length;

            Dimensions = new long[N];
            Multiplied = new long[N + 1];
            Multiplied[N] = 1;
            for (int i = N - 1; i >= 0; i--)
            {
                Dimensions[i] = dims[i];
                Multiplied[i] = Dimensions[i] * Multiplied[i + 1];
            }
        }

        public long this[int x]
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
        public long Index(params int[] dims)
        {
            long res = 0;
            for (int i = 0; i < dims.Length; i++)
                res += dims[i] * Multiplied[i + 1];
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Shape Clone()
        {
            Shape s = Shape.DimensionOf(this.N);
            for (int i = 0; i < s.N; i++)
            {
                s.Dimensions[i] = this.Dimensions[i];
                s.Multiplied[i] = this.Multiplied[i];
            }
            s.Multiplied[this.N] = 1;
            return s;
        }

        public static bool operator ==(Shape obj, object obj2)
        {
            return obj.Equals(obj2);
        }

        public static bool operator !=(Shape obj, object obj2)
        {
            return !obj.Equals(obj2);
        }

        public static bool operator ==(Shape obj, Shape obj2)
        {
            return obj.EqualShape(obj2);
        }

        public static bool operator !=(Shape obj, Shape obj2)
        {
            return !obj.EqualShape(obj2);
        }

        public override bool Equals(object obj)
        {
            if (obj != null && obj is Shape s)
            {
                return EqualShape(s);
            }
            else
                return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private bool EqualShape(Shape s)
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
            Shape c = Shape.DimensionOf(s1.N + s2.N);
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
            Shape c = Shape.DimensionOf(s1.N - s2.N + s3.N);

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

            Shape res = Shape.DimensionOf(s1.N);
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

            Shape res = Shape.DimensionOf(s1.N);
            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s1[i] * s2[i];
            res.CalculateMultiplied();

            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Shape RemoveLastDimension(Shape s)
        {
            Shape res = Shape.DimensionOf(s.N - 1);

            for (int i = 0; i < res.N; i++)
                res.Dimensions[i] = s.Dimensions[i];

            res.CalculateMultiplied();

            return res;
        }


    
    }
}
