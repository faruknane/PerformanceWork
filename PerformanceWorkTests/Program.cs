using PerformanceWork;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Index = PerformanceWork.OptimizedNumerics.Index;

namespace PerformanceWorkTests
{
    public unsafe class Program
    {
        public static int Size = 5000;

        private static void MatrixMultiply()
        {
            //Vectorization.MatrixMultiply(ref a, ref b, ref c);
            //MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.NoTrans, a.D1, b.D2, b.D1, 1.0f, a.GetPointer(), b.D1, b.GetPointer(), b.D2, 0.0f, c.GetPointer(), b.D2);
        }

        public static bool ArrayEqual<T>(T[] v1, T[] v2)
        {
            if (v1.Length != v2.Length) return false;
            for (int i = 0; i < v1.Length; i++)
                if (!v1[i].Equals(v2[i]))
                    return false;
            return true;
        }

        static unsafe void Main(string[] args)
        {
            Shape s = Shape.NewShape(2, 3, 5, 7);
            Index a = Index.NewIndex(s);
            a.SetZero();

            for (int i = 0; i < s.TotalSize; i++)
            {
                Console.WriteLine(a);
                a.Add(1);
            }
            //for (int i = 0; i < 100; i++)
            //{
            //    Index index = Index.NewIndex(s1);
            //    index.Indexes[0] = i;
            //    index.Indexes[1] = i + 1;
            //    Console.WriteLine($"{index[0]}, {index[1]}");
            //    Index.Return(index);
            //}
            //Shape.Return(s1);

            //Console.WriteLine(Index.ObjectPool.Count);
            //Console.WriteLine(Index.ObjectPool.UnreturnedCount);
            //Console.WriteLine(Index.IndexPool.UnreturnedArrayCount);


            //int size = 1000;
            //Tensor<float> t = new Tensor<float>((size, size), false, 0);
            //Tensor<float> t2 = new Tensor<float>((size, size), false, 0);

            //for (int i = 0; i < t.Shape[0]; i++)
            //    for (int j = 0; j < t.Shape[1]; j++)
            //    {
            //        ((float*)t.Array)[t.Shape.Index(i, j)] = j;
            //        ((float*)t2.Array)[t2.Shape.Index(i, j)] = i;
            //    }

            //Stopwatch s = new Stopwatch();
            //s.Start();
            //for (int i = 0; i < 50; i++)
            //{
            //    var t3 = Tensor<float>.Sum(t, t2);
            //    t3.Dispose();
            //}
            //s.Stop();
            //Console.WriteLine(s.ElapsedMilliseconds);
            //t.Dispose();    
            //t2.Dispose();
            //Console.WriteLine(Tensor<float>.Host.UnreturnedArrayCount);
            ////Tensor<float>.GetDevicePool(0).ClearMemory();
            //return;

        }


    }
}
