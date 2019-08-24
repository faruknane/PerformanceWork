using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using PerformanceWork;
using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading;

namespace PerformanceWorkTests
{
    public class Program
    {
        public static int Size = 14 * 14 * 5 * 5;

        public unsafe static void VectorizationAddAVX()
        {
            Vectorization.AddAVX(a.Array, b.Array, c.Array, c.Array.Length);
        }

        public unsafe static void DotProductAVX()
        {
            Vectorization.DotProductAVX(ref a.Array, ref b.Array, c.Array.Length);
        }
        public unsafe static void DotProductAVXParallel()
        {
            fixed (float* ptra = a.Array, ptrb = b.Array)
            {
                Vectorization.DotProductAVXParallel(ptra, ptrb, a.Array.Length);
            }
        }
        public unsafe static void MatrixMultiply()
        {
            Vectorization.MatrixMultiply(ref a, ref b, ref c);
        }


        public static Matrix a, b, c;

        static unsafe void Main(string[] args)
        {
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

            for(int i =0; i < 8; i++)
            {
                a.Array[i] = i;
                b.Array[i] = i;
            }
            fixed (float* ptr_a = a.Array, ptr_b = b.Array)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[0]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[0]);
                Vector256<float> res = Avx2.Compare(v1, v2, FloatComparisonMode.UnorderedEqualNonSignaling);
                Console.WriteLine(res);
            }


            //Stopwatch s = new Stopwatch();
            

            //s.Restart();
            //for (int i = 0; i < 1; i++)
            //    Program.MatrixMultiply();
            //s.Stop();
            //long time = s.ElapsedMilliseconds;
            //Console.WriteLine("MatrixMultiply Size: " + Size + ", Time: " + time);

            //a.Dispose();
            //b.Dispose();
            //c.Dispose();


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
