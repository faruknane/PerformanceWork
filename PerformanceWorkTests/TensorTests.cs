using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork;
using PerformanceWork.DeepLearning.Kernels.Cpu;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace PerformanceWorkTests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public unsafe void Reshape()
        {
            float[] a = new float[] { 1, 2, 3, 4, 5, 6 };

            Tensor x1 = a.ToDisposedTensor();
            Tensor x2 = x1.Reshape(3, 2);

            Assert.AreEqual(x1.Base, x2.Base);
            
            fixed(float* ptr = a)
                Assert.AreEqual((long)x1.Base.Array, (long)ptr);

            Console.WriteLine(x1);
            Console.WriteLine(x2);
        }
        


    }


    [TestClass]
    public class ShapeTests
    {

        [TestMethod]
        public void Equality()
        {
            Shape a = new Shape(1, 2, 3);
            Shape b = new Shape(1, 2, 3);
            Assert.AreEqual(a, b);
            Assert.IsTrue(a == b);

            b.Dimensions[1] = 5;
            b.CalculateMultiplied();

            Assert.IsFalse(a == b);
            Assert.AreNotEqual(a, b);
        }


        [TestMethod]
        public void ShapeCombining()
        {
            Shape s1 = new Shape(3, 2);
            Shape s2 = new Shape(5, 6, 7);
            Shape res = Shape.Combine(s1, s2);
            Shape res2 = new Shape(3, 2, 5, 6, 7);

            Assert.AreEqual(res, res2);
            Assert.IsTrue(res == res2);
        }


        [TestMethod]
        public void ShapeSwapTail()
        {
            Shape s1 = new Shape(3, 5, 6, 7, 2);
            Shape s2 = new Shape(7, 2);
            Shape s3 = new Shape(1, 1);
            Shape res = Shape.SwapTail(s1, s2, s3);
            Shape res2 = new Shape(3, 5, 6, 1, 1);

            Assert.AreEqual(res, res2);
            Assert.IsTrue(res == res2);
        }

    }
}
