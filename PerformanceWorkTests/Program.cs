using PerformanceWork;
using PerformanceWork.DeepLearning.Kernels.Cpu;
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
            float[] a = new[] { 1f, -10f, 5f, 3.2f, -5.5f, -3.9f, -12.3f, 4.1f };
            float[] res = new float[a.Length];

            fixed(float* ptr_data = a, ptr_res = res)
            {
                VectorizationFloat.Softmax(ptr_data, ptr_res, a.Length, a.Length);
                Console.WriteLine(float.PositiveInfinity);
            }
            //Console.WriteLine(Tensor<float>.Host.UnreturnedArrayCount);
            ////Tensor<float>.GetDevicePool(0).ClearMemory();
            //return;

        }


    }
}
