using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork;
using PerformanceWork.DeepLearning.Kernels.NvidiaGpu;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        public unsafe void AddFloat32_Test()
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
        public unsafe void AddFloat32_Test_2()
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
        public unsafe void AddFloat32_Test_3()
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
        public unsafe void MultiplyFloat32_Test()
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

        [TestMethod]
        public unsafe void Einsum1()
        {
            float[,] fA = new float[2, 3] { { 1, 2, 3 }, { 4, 5, 6 } };
            float[,] fB = new float[3, 2] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            float[,] fRes = new float[2, 2] { { 22, 28 }, { 49, 64 } };

            Tensor A = fA.ToDisposedTensor().CopyTo(Device.Nvidia(0));
            Tensor B = fB.ToDisposedTensor().CopyTo(Device.Nvidia(0));

            Tensor C = new Tensor((2, 2), TensorConfig.NvidiaGPU_Float32);


            /* can use instead -> NvidiaGpuKernels.Einsum(C, A, "mk", B, "kn", C, "mn");
             * 
             * C = A*B + C
             * A -> mk
             * B -> kn
             * C -> mn
             */
                
            NvidiaGpuKernels.Einsum(C, "mk,kn->mn", A, B, C);
          

            Tensor myres = C.CopyTo(Device.Host);
            Tensor expectedres = fRes.ToDisposedTensor();

            Assert.AreEqual(myres.ToString(), expectedres.ToString());
            //there is no equailty operator for tensors so we use ToString for now!

            A.Dispose();
            B.Dispose();
            C.Dispose();
            myres.Dispose();
        }

        [TestMethod]
        public unsafe void Einsum2()
        {
            float[,] fA = new float[2, 3] { { 1, 2, 3 }, { 4, 5, 6 } };
            float[,] fB = new float[3, 2] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            float[,] fRes = new float[3, 2] { { 1, 8 }, { 6, 20 }, { 15, 36 } };

            Tensor expectedres = fRes.ToDisposedTensor();

            Tensor A = fA.ToDisposedTensor().CopyTo(Device.Nvidia(0));
            Tensor B = fB.ToDisposedTensor().CopyTo(Device.Nvidia(0));

            Tensor C = new Tensor(expectedres.Shape.Clone(), TensorConfig.NvidiaGPU_Float32);

            NvidiaGpuKernels.Einsum(C, "mk,km->km", A, B, C);

            Tensor myres = C.CopyTo(Device.Host);

            Assert.AreEqual(expectedres.ToString(), myres.ToString());
            //there is no equailty operator for tensors so we use ToString for now!

            A.Dispose();
            B.Dispose();
            C.Dispose();
            myres.Dispose();
        }

        [TestMethod]
        public unsafe void Einsum3()
        {
            float[,,,] fA = new float[11, 1, 5, 3];
            float[,,,] fB = new float[1, 7, 3, 2];


            Tensor A = fA.ToDisposedTensor().CopyTo(Device.Nvidia(0));
            Tensor B = fB.ToDisposedTensor().CopyTo(Device.Nvidia(0));

            Tensor C = new Tensor(new Shape(11, 7, 5, 2), TensorConfig.NvidiaGPU_Float32);

            NvidiaGpuKernels.Einsum(C, "a.ij,.bjk->abik", A, B, C);

            Tensor myres = C.CopyTo(Device.Host);

            A.Dispose();
            B.Dispose();
            C.Dispose();
            myres.Dispose();
        }


        [TestMethod]
        public unsafe void Einsum4()
        {
            Tensor A = Tensor.Arange(0, 6).Reshape(2, 3).CopyTo(Device.Nvidia(0));
            Tensor B = Tensor.Arange(0, 6).Reshape(2, 3).CopyTo(Device.Nvidia(0));
            Tensor C = new Tensor(new Shape(2, 3), TensorConfig.NvidiaGPU_Float32);

            NvidiaGpuKernels.Einsum(C, "mn,mn->mn", A, B, C);

            Tensor myres = C.CopyTo(Device.Host);

            Console.WriteLine(myres);
            Assert.IsTrue("[[0, 1, 4][9, 16, 25]]" == myres.ToString());

            A.Dispose();
            B.Dispose();
            C.Dispose();
            myres.Dispose();
        }
    }
}
