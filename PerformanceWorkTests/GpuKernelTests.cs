using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork;
using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Linq;
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

    }
}
