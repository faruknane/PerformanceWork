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
        public unsafe void AddFloat32NvidiaGpuBig()
        {
            int size = 1000000;
            Tensor a = new Tensor(new Shape(size), TensorConfig.NvidiaGPU_Float32);
            Tensor b = new Tensor(new Shape(size), TensorConfig.NvidiaGPU_Float32);
            Tensor myres = NvidiaGpuKernels.AddFloat32(a, b);
            myres.Dispose();
            a.Dispose();
            b.Dispose();
        }

        [TestMethod]
        public unsafe void AddKernelFloat32()
        {
            Random r = new Random();
            int size = 10000;
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

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));


            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));
            t2 = x2.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));
            Tensor myres = NvidiaGpuKernels.AddFloat32(t1, t2);


            //Console.WriteLine(myres);
            //Console.WriteLine(expected_res);
            Assert.AreEqual(myres.ToString(), expected_res.ToString());
            myres.Dispose();
            t1.Dispose();
            t2.Dispose();
        }

        [TestMethod]
        public unsafe void AddKernelFloat32_2()
        {
            Random r = new Random();
            int size = 10000;
            int size2 = 1000;
            float[] x1, x2, expres;
            x1 = new float[size];
            x2 = new float[size2];
            expres = new float[size];
            for (int i = 0; i < size; i++)
            {
                x1[i] = (float)r.NextDouble() * 10 - 5;
                if (i < size2)
                    x2[i] = (float)r.NextDouble() * 10 - 5;
                expres[i] = x1[i] + x2[i % size2];
            }

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));


            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));
            t2 = x2.ToDisposedTensor(new Shape(size2)).CopyTo(Device.Nvidia(0));
            Tensor myres = NvidiaGpuKernels.AddFloat32(t1, t2);


            //Console.WriteLine(myres);
            //Console.WriteLine(expected_res);
            Assert.AreEqual(myres.ToString(), expected_res.ToString());
            myres.Dispose();
            t1.Dispose();
            t2.Dispose();
        }

        [TestMethod]
        public unsafe void AddKernelFloat32_3()
        {
            const int size = 10;
            const int size2 = 5;
            float[] x1, x2, expres;
            x1 = new float[size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            x2 = new float[size2] { 1, 2, 3, 4, 5 };
            expres = new float[size];

            for (int i = 0; i < size; i++)
                expres[i] = x1[i] + x2[i % size2];

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));

            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));
            t2 = x2.ToDisposedTensor(new Shape(size2)).CopyTo(Device.Nvidia(0));
            Tensor myres = NvidiaGpuKernels.AddFloat32(t1, t2);


            Console.WriteLine(myres);
            Console.WriteLine(expected_res);
            Assert.AreEqual(myres.ToString(), expected_res.ToString());
            myres.Dispose();
            t1.Dispose();
            t2.Dispose();
        }

        [TestMethod]
        public unsafe void MultiplyKernelFloat32()
        {
            const int size = 10;
            const int size2 = 5;
            float[] x1, x2, expres;
            x1 = new float[size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            x2 = new float[size2] { 1, 2, 3, 4, 5 };
            expres = new float[size];

            for (int i = 0; i < size; i++)
                expres[i] = x1[i] * x2[i % size2];

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));

            Tensor t1, t2;
            t1 = x1.ToDisposedTensor().CopyTo(Device.Nvidia(0));
            t2 = x2.ToDisposedTensor().CopyTo(Device.Nvidia(0));
            Tensor myres = NvidiaGpuKernels.MultiplyFloat32(t1, t2);


            Console.WriteLine(myres);
            Console.WriteLine(expected_res);
            Assert.AreEqual(myres.ToString(), expected_res.ToString());
            myres.Dispose();
            t1.Dispose();
            t2.Dispose();
        }

        [TestMethod]
        public unsafe void DivideKernelFloat32()
        {
            const int size = 10;
            const int size2 = 5;
            float[] x1, x2, expres;
            x1 = new float[size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            x2 = new float[size2] { 1, 2, 3, 4, 5 };
            expres = new float[size];

            for (int i = 0; i < size; i++)
                expres[i] = x1[i] / x2[i % size2];

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));

            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size)).CopyTo(Device.Nvidia(0));
            t2 = x2.ToDisposedTensor(new Shape(size2)).CopyTo(Device.Nvidia(0));
            Tensor myres = NvidiaGpuKernels.DivideFloat32(t1, t2);


            Console.WriteLine(myres);
            Console.WriteLine(expected_res);
            Assert.AreEqual(myres.ToString(), expected_res.ToString());
            myres.Dispose();
            t1.Dispose();
            t2.Dispose();
        }

    }
}
