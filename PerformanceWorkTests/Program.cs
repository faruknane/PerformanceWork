using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using PerformanceWork;
using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading;

namespace PerformanceWorkTests
{
    public unsafe class Program
    {
        public static int Size = 1000;


        private static void MatrixMultiply()
        {
            //Vectorization.MatrixMultiply(ref a, ref b, ref c);
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.NoTrans, a.D1, b.D2, b.D1, 1.0f, a.GetPointer(), b.D1, b.GetPointer(), b.D2, 0.0f, c.GetPointer(), b.D2);
        }

        public static Matrix a, b, c;
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
            a = new Matrix(5, 2);
            b = new Matrix(5, 3);
            c = new Matrix(2, 3);

            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 2; j++)
                    a[i, j] = i+1;

            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++)
                    b[i, j] = j+1;

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 2; j++)
                    Console.Write(a[i, j] + ", ");
                Console.Write("\n");
            }
            Console.Write("\n");

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 3; j++)
                    Console.Write(b[i, j] + ", ");
                Console.Write("\n");
            }
            Vectorization.TransposeAandMatrixMultiply(a.Array, a.D1, a.D2, b.Array, b.D1, b.D2, c.Array);
            Console.Write("\n");

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                    Console.Write(c[i, j] + ", ");
                Console.Write("\n");
            }
            return;
            a = new Matrix(Size, Size);
            b = new Matrix(Size, Size);
            c = new Matrix(Size, Size);
            for (int i = 0; i < Size; i++)
                for (int j = 0; j < Size; j++)
                    a[i, j] = i;
            for (int i = 0; i < Size; i++)
                for (int j = 0; j < Size; j++)
                    b[i, j] = j;
            for (int i = 0; i < Size; i++)
                for (int j = 0; j < Size; j++)
                    c[i, j] = 0;

            //for (int i = 0; i < 10; i++)
            //{
            //    Vectorization.DotProductFMA(a.Array, b.Array, a.Array.Length);
            //}

            Stopwatch s = new Stopwatch();

            s.Restart();
            for (int i = 0; i < 100; i++)
            {
                MatrixMultiply();
            }
            s.Stop();
            long time = s.ElapsedMilliseconds;
            Console.WriteLine("DotProductFMA Size: " + Size + ", Time: " + time);

            a.Dispose();
            b.Dispose();
            c.Dispose();


            //int time = 0;
            //    Stopwatch s = new Stopwatch();
            //    s.Start();
            //for (int i = 0; i < 10; i++)
            //{
            //    p.DotProductAVXParallel();
            //}
            //s.Stop();
            //Console.WriteLine(s.ElapsedMilliseconds);
            //var summary = BenchmarkRunner.Run<Program>();
        }

       
    }
}
