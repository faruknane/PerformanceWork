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
            float[] vdata = new float[] { 0.1f, -0.2f, 2, 3, -1, 2, -5, -10, 9, -8 };
            float[] graddata = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };

            fixed (float* ptr_v = vdata, ptr_grad = graddata)
            {
                Tensor v = Tensor.LoadFloatArrayToTensorHost(ptr_v, 0, vdata.Length, Shape.NewShape(vdata.Length));
                Tensor grad = Tensor.LoadFloatArrayToTensorHost(ptr_grad, 0, graddata.Length, Shape.NewShape(graddata.Length));

                Tensor reluv = CpuKernels.ReluFloat(v);
                Tensor relugrad = CpuKernels.ReluFloat_GetGradient_0(grad, v);

                Console.WriteLine("v: " + v);
                Console.WriteLine("reluv: " + reluv);
                Console.WriteLine("grad: " + grad);
                Console.WriteLine("relugrad: " + relugrad);

            }

            //Console.WriteLine(Tensor<float>.Host.UnreturnedArrayCount);
            ////Tensor<float>.GetDevicePool(0).ClearMemory();
            //return;

        }


    }
}
