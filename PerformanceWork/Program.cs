using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using PerformanceWork.OptimizedNumerics;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace PerformanceWork
{
    public class Program
    {
        public static int Size = 14;

        public unsafe static void VectorizationAddAVX()
        {
            Vectorization.AddAVX(ref a.Array, ref b.Array, ref c.Array, c.Array.Length);
        }

        public unsafe static void DotProductAVX()
        {
            Vectorization.DotProductAVX(ref a.Array, ref b.Array, c.Array.Length);
        }
        public unsafe static void DotProductAVXParallel()
        {
            fixed(float* ptra = a.Array, ptrb = b.Array)
            {
                Vectorization.DotProductAVXParallel(ptra, ptrb, a.Array.Length);
            }
        }
        public unsafe static void MatrixMultiply()
        {
            Vectorization.MatrixMultiply(ref a, ref b, ref c);
        }
        public unsafe static void MatrixMultiplyWithTranspose()
        {
            Vectorization.MatrixMultiplyWithTranspose(ref a, ref b, ref c);
        }

        public static Matrix a, b, c;

        static unsafe void Main(string[] args)
        {
            Stopwatch s = new Stopwatch();
            for (int scale = 0; scale < 9; scale++)
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

                Program.MatrixMultiply();
                s.Restart();
                for(int i = 0; i < 10; i++)
                    Program.MatrixMultiply();
                s.Stop();
                long time = s.ElapsedMilliseconds;
                Console.WriteLine("MatrixMultiply Size: " + Size + ", Time: " + time);

                Size *= 2;
                a.Dispose();
                b.Dispose();
                c.Dispose();
            }
            

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
