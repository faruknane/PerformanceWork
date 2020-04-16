using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Index : IDisposable
    {
        public static ArrayPool ArrayPool = ArrayPool.Create(10, 10);
        public static ObjectPool<Index> ObjectPool = new ObjectPool<Index>();

        //todo add + - operators with tuples in order to let users make rnn more simple like time - (1, 1) time * (1,2)

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Index NewIndex(Shape shape)
        {
            Index i = ObjectPool.Rent();
            if (i == null)
                i = new Index(shape);
            else
            {
                i.Shape = shape;
                i.ArrayReturned = false;
                i.N = shape.N;
                i.Indices = (int*)ArrayPool.Rent(i.N, out i.Length1, DataType.Type.Int32);
            }
            return i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Return(Index i)
        {
            if (!i.ArrayReturned)
                i.Dispose();
            ObjectPool.Return(i);
        }

        public int* Indices;
        public int N { get; private set; }
        public Shape Shape { get; private set; }

        private int Length1;
        private bool ArrayReturned = false;

        public int this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get => Indices[x];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Index(Shape s)
        {
            this.Shape = s;
            this.N = s.N;
            Indices = (int*)ArrayPool.Rent(this.N, out Length1, DataType.Type.Int32);
        }

        public void Add(int x)
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

        public void SetZero()
        {
            for (int i = 0; i < N; i++)
                Indices[i] = 0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            if (ArrayReturned)
                throw new Exception("The index object is already returned!");
            ArrayReturned = true;

            ArrayPool.Return(Indices, Length1);
            GC.SuppressFinalize(this);
        }

        public override string ToString()
        {
            string res = "";
            for (int i = 0; i < N; i++)
                res += Indices[i] + ", ";
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        ~Index()
        {
            if (!this.ArrayReturned)
            {
                this.Dispose();
            }
        }
    }
}
