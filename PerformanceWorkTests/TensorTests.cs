using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork;
using PerformanceWork.DeepLearning.Kernels.Cpu;
using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Text;

namespace PerformanceWorkTests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public void SumTensorsCPU()
        {
            Tensor t = new Tensor((3, 5), DataType.Type.Float, DeviceIndicator.Host());
            Tensor t2 = new Tensor((3, 5), DataType.Type.Float, DeviceIndicator.Host());
            unsafe
            {
                for (int i = 0; i < t.Shape[0]; i++)
                    for (int j = 0; j < t.Shape[1]; j++)
                    {
                        ((float*)t.Array)[t.Shape.Index(i, j)] = j;
                        ((float*)t2.Array)[t2.Shape.Index(i, j)] = i;
                    }

                Tensor t3 = Tensor.Sum(t, t2);

                for (int i = 0; i < t.Shape[0]; i++)
                    for (int j = 0; j < t.Shape[1]; j++)
                        Assert.AreEqual(((float*)t3.Array)[t3.Shape.Index(i, j)], i + j);

                t.Dispose();
                t2.Dispose();
                t3.Dispose();
            }
        }

        [TestMethod]
        public unsafe void MatrixMultiply()
        {

            for (int kk = 0; kk < 10; kk++)
            {
                Random r = new Random();

                int n = r.Next(2, 30);
                int m = r.Next(2, 30);
                int p = r.Next(2, 30);

                Tensor a = new Tensor((n, m), DataType.Type.Float, DeviceIndicator.Host());
                Tensor b = new Tensor((m, p), DataType.Type.Float, DeviceIndicator.Host());
                Tensor res = new Tensor((n, p), DataType.Type.Float, DeviceIndicator.Host());

                res.SetFloat(0);

                for (int i = 0; i < a.Shape[0]; i++)
                    for (int j = 0; j < a.Shape[1]; j++)
                        ((float*)a.Array)[a.Shape.Index(i, j)] = r.Next(-10, 10);

                for (int i = 0; i < b.Shape[0]; i++)
                    for (int j = 0; j < b.Shape[1]; j++)
                        ((float*)b.Array)[b.Shape.Index(i, j)] = r.Next(-10, 10);

                for (int i = 0; i < a.Shape[0]; i++)
                    for (int j = 0; j < a.Shape[1]; j++)
                        for (int k = 0; k < b.Shape[1]; k++)
                            ((float*)res.Array)[res.Shape.Index(i, k)] += ((float*)a.Array)[a.Shape.Index(i, j)] * ((float*)b.Array)[b.Shape.Index(j, k)];

                Tensor c = Tensor.MatrixMultiply(a, b);

                if (!VectorizationFloat.ElementWiseIsEqualsAVX((float*)res.Array, (float*)c.Array, res.Shape.TotalSize))
                {
                    throw new Exception("Eşit Değil!");
                }
                a.Dispose();
                b.Dispose();
                c.Dispose();
                res.Dispose();
            }
        }


        [TestMethod]
        public unsafe void ShapeCombining()
        {
            Shape s1 = new Shape(3, 2);
            Shape s2 = new Shape(5, 6, 7);
            Shape res = Shape.Combine(s1, s2);
            Shape res2 = new Shape(3, 2, 5, 6, 7);

            for (int i = 0; i < res.N; i++)
                if (res.Dimensions[i] != res2.Dimensions[i])
                    throw new Exception("Hata");
        }


        [TestMethod]
        public unsafe void ShapeSwapTail()
        {
            Shape s1 = new Shape(3, 5, 6, 7, 2);
            Shape s2 = new Shape(7, 2);
            Shape s3 = new Shape(1, 1);
            Shape res = Shape.SwapTail(s1, s2, s3);
            Shape res2 = new Shape(3, 5, 6, 1, 1);

            for (int i = 0; i < res.N; i++)
                if (res.Dimensions[i] != res2.Dimensions[i])
                    throw new Exception("Hata");
        }


        [TestMethod]
        public unsafe void ReluFloatCpu()
        {
            float[] vdata = new float[] { 0.1f, -0.2f, 2, 3, -1, 2, -5, -10, 9, -8 };
            float[] graddata = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };

            fixed (float* ptr_v = vdata, ptr_grad = graddata)
            {
                Tensor v = Tensor.LoadFloatArray(vdata, new Shape((vdata.Length, true)));
                Tensor grad = Tensor.LoadFloatArray(graddata, new Shape((graddata.Length, true)));

                Tensor reluv = CpuKernels.ReluFloat(v);
                Tensor relugrad = CpuKernels.ReluFloat_GetGradient_0(grad, v);

                Console.WriteLine(v);
                Console.WriteLine(grad);

                Console.WriteLine(reluv);
                Console.WriteLine(relugrad);
                //todo make checks

            }
        }
    }
}
