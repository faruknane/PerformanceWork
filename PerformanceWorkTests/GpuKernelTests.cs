using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork;
using PerformanceWork.DeepLearning.Kernels.NvidiaGpu;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWorkTests
{
    [TestClass]
    public class GpuKernelTests
    {
        [TestMethod]
        public unsafe void CopyCpuToNvidiaGpuSmall()
        {
            const int size = 10;
            var f = new float[size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            Tensor a = f.ToDisposedTensor(new Shape(size), NumberType.Float32);
            Tensor b = Tensor.CopyTo(a, Device.Nvidia(0));
            Console.WriteLine(b);
            b.Dispose();
        }

        [TestMethod]
        public unsafe void CopyCpuToNvidiaGpuBig()
        {
            int size = 10000000;
            var f = new float[size];
            Tensor a = f.ToDisposedTensor(new Shape(size), NumberType.Float32);
            Tensor b = Tensor.CopyTo(a, Device.Nvidia(0));
            //Console.WriteLine(b);
            b.Dispose();
        }

        [TestMethod]
        public unsafe void AddKernelFloat32()
        {
            Random r = new Random();
            int size = r.Next(10000000, 20000000);
            float[] x1, x2, expres;
            x1 = new float[size];
            x2 = new float[size];
            expres = new float[size];
            for (int i = 0; i < size; i++)
            {
                x1[i] = (float)r.NextDouble() * 10 - 5;
                x2[i] = (float)r.NextDouble() * 10 - 5;
                expres[i] = x1[i] + x2[i];
            }

            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));
            t2 = x2.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));

            Tensor myres = NvidiaGpuKernels.AddFloat32(t1, t2);
            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));

            //Console.WriteLine(myres);
            //Console.WriteLine(expected_res);
            //Assert.AreEqual(myres.ToString(), expected_res.ToString());

            t1.Dispose();
            t2.Dispose();
        }

    }
}
