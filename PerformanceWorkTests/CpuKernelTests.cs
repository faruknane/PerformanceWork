using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork.DeepLearning.Kernels.Cpu;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWorkTests
{
    [TestClass]
    public class CpuKernelTests
    {
        [TestMethod]
        public unsafe void AddFloat32_Test()
        {
            const int size = 11;
            const int size2 = 5;
            float[] x1, x2, expres;
            x1 = new float[size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            x2 = new float[size2] { 1, 2, 3, 4, 5 };
            expres = new float[size];

            for (int i = 0; i < size; i++)
                expres[i] = x1[i] + x2[i % size2];

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));

            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size));
            t2 = x2.ToDisposedTensor(new Shape(size2));
            Tensor myres = CpuKernels.AddFloat32(t1, t2);


            Console.WriteLine(myres);
            Console.WriteLine(expected_res);
            Assert.AreEqual(expected_res.ToString(), myres.ToString());
            myres.Dispose();
        }

        [TestMethod]
        public unsafe void SubtractFloat32_Test()
        {
            const int size = 11;
            const int size2 = 5;
            float[] x1, x2, expres;
            x1 = new float[size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            x2 = new float[size2] { 1, 2, 3, 4, 5 };
            expres = new float[size];

            for (int i = 0; i < size; i++)
                expres[i] = x1[i] - x2[i % size2];

            Tensor expected_res = expres.ToDisposedTensor(new Shape(size));

            Tensor t1, t2;
            t1 = x1.ToDisposedTensor(new Shape(size));
            t2 = x2.ToDisposedTensor(new Shape(size2));
            Tensor myres = CpuKernels.SubtractFloat32(t1, t2);


            Console.WriteLine(myres);
            Console.WriteLine(expected_res);
            Assert.AreEqual(expected_res.ToString(), myres.ToString());
            myres.Dispose();
        }

        public (float[], Tensor) CreateRandomTensor(int size = 9)
        {
            Random random = new Random();
            float[] arr = new float[size];
            for (int i = 0; i < size; i++)
            {
                arr[i] = (float)random.NextDouble();
            }
            Tensor tensor = Tensor.Clone(Tensor.ToDisposedTensor(arr, new Shape(size), NumberType.Float32));
            return (arr, tensor);
        }

        [TestMethod]
        public void Add()
        {
            Tensor expected = new Tensor(new Shape(3, 3), TensorConfig.Host_Float32);
            expected.SetValue(15);
            Tensor calculated = new Tensor(new Shape(3, 3), TensorConfig.Host_Float32);
            calculated.SetValue(0);
            Tensor[] inputs = new Tensor[5];
            for (int i = 0; i < 5; i++)
            {
                inputs[i] = new Tensor(new Shape(3, 3), TensorConfig.Host_Float32);
                inputs[i].SetValue(3);
            }
            CpuKernels.AddFloat32(calculated, inputs);
            Assert.AreEqual(expected.ToString(), calculated.ToString());
            Tensor calculated2 = CpuKernels.AddFloat32(inputs);
            Assert.AreEqual(expected.ToString(), calculated2.ToString());

            expected.Dispose();
            calculated.Dispose();
            for (int i = 0; i < 5; i++)
                inputs[i].Dispose();
        }

        [TestMethod]
        public void AddRandom()
        {
            int arrSize = 5;
            int tensorSize = 9;

            // initialize test tensors
            Tensor expected, calculated;
            float[] expectedArr = new float[tensorSize];
            (_, calculated) = CreateRandomTensor(tensorSize);
            Tensor[] inputs = new Tensor[arrSize];
            float[][] arrays = new float[arrSize][];
            for (int i = 0; i < arrSize; i++)
            {
                (arrays[i], inputs[i]) = CreateRandomTensor(tensorSize);
                for (int j = 0; j < tensorSize; j++)
                {
                    expectedArr[j] += arrays[i][j];
                }
            }

            // add tensors
            expected = Tensor.ToDisposedTensor(expectedArr, new Shape(tensorSize), NumberType.Float32);


            CpuKernels.AddFloat32(calculated, inputs);

            Console.WriteLine(calculated.ToString());
            Console.WriteLine(expected.ToString());

            Assert.AreEqual(calculated.ToString(), expected.ToString());

            //need to dipose tensors to remove them from memory
            //we don't need to do this manually but better doing
            calculated.Dispose();
            for (int i = 0; i < arrSize; i++)
                inputs[i].Dispose();

            //expected.Dispose(); don't dispose expected tensor because it is already disposed tensor. line 72

        }

        [TestMethod]
        public void MultiplyRandom()
        {
            Tensor m1, m2, calculated, result;
            float[] arr1, arr2, arr_expected;
            (arr1, m1) = CreateRandomTensor();
            (arr2, m2) = CreateRandomTensor();
            arr_expected = new float[9];

            //CpuKernels.MultiplyFloat32 already uses ElementWiseMultiplyAVX method. We should calculate the results manually.
            for (int i = 0; i < arr_expected.Length; i++)
                arr_expected[i] = arr1[i] * arr2[i];

            //VectorizationFloat.ElementWiseMultiplyAVX(arr1, arr2, arr_expected, 9);

            result = Tensor.ToDisposedTensor(arr_expected, new Shape(9), NumberType.Float32);

            calculated = CpuKernels.MultiplyFloat32(m1, m2);

            Console.WriteLine(calculated.ToString());
            Console.WriteLine(result.ToString());

            Assert.AreEqual(calculated.ToString(), result.ToString());

            m1.Dispose();
            m2.Dispose();
            calculated.Dispose();
        }

        [TestMethod]
        public unsafe void ReluFloatCpu()
        {
            float[] vdata = new float[] { 0.1f, -0.2f, 2, 3, -1, 2, -5, -10, 9, -8 };
            float[] graddata = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };

            fixed (float* ptr_v = vdata, ptr_grad = graddata)
            {
                Tensor v = Tensor.ToDisposedTensor(vdata, new Shape(vdata.Length), NumberType.Float32);
                Tensor grad = Tensor.ToDisposedTensor(graddata, new Shape(graddata.Length), NumberType.Float32);

                Tensor reluv = CpuKernels.ReluFloat32(v);
                Tensor relugrad = CpuKernels.ReluFloat32_GetGradient_0(grad, v);

                Console.WriteLine(v);
                Console.WriteLine(grad);

                Console.WriteLine(reluv);
                Console.WriteLine(relugrad);
                //todo make checks

                reluv.Dispose();
                relugrad.Dispose();
            }
        }
    }
}
