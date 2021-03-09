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


        


    }
}
