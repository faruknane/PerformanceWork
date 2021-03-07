using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Index
    {
        public long[] Indices;
        public int N { get; private set; }
        public Shape Shape { get; private set; }

        public long this[int x]
        {
            get => Indices[x];
            set => Indices[x] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Index(Shape s)
        {
            this.Shape = s;
            this.N = s.N;
            Indices = new long[N];
        }

        public void Increase(int x)
        {
            //no error handling for performance issues
            int i = N - 1;
            Indices[i] += x;
            while (Indices[i] >= Shape.Dimensions[i] && i > 0)
            {
                Indices[i - 1] += Indices[i] / Shape.Dimensions[i];
                Indices[i] %= Shape.Dimensions[i];
                i--;
            }
        }

        public void Add(Shape x)
        {
            //no error handling for performance issues
            for (int i = 0; i < this.N; i++)
                Indices[i] += x[i];
        }

        public void Subtract(Shape x)
        {
            //no error handling for performance issues
            for (int i = 0; i < this.N; i++)
                Indices[i] -= x[i];
        }

        public void SetZero()
        {
            for (int i = 0; i < N; i++)
                Indices[i] = 0;
        }

        public override string ToString()
        {
            string res = "";
            for (int i = 0; i < N; i++)
                res += Indices[i] + ", ";
            return res;
        }

        public static Index operator +(Index a, int b)
        {
            Index i = a.Clone();
            i[i.N - 1] += b;
            return i;
        }

        public static Index operator -(Index a, int b)
        {
            Index i = a.Clone();
            i[i.N - 1] -= b;
            return i;
        }

        public static Index operator +(Index a, Shape b)
        {
            Index i = a.Clone();
            i.Add(b);
            return i;
        }

        public static Index operator -(Index a, Shape b)
        {
            Index i = a.Clone();
            i.Subtract(b);
            return i;
        }

        public Index Clone()
        {
            Index i = new Index(this.Shape);
            i.N = this.N;
            i.Indices = new long[i.N];
            for (int j = 0; j < i.N; j++)
                i[j] = this[j];
            return i;
        }
    }
}
