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

namespace PerformanceWorkTests
{
    public unsafe class Program
    {
        static void f()
        {
            for (int i = 0; i < 100; i++)
            {
                //4mb
                Tensor a = new Tensor(new Shape(1024 * 1024), DeviceConfig.Host_Float32);
                a.Dispose();
            }
        }

        static unsafe void Main(string[] args)
        {
            f();
            Console.WriteLine(Tensor.DisposedCount);
            TensorPool.GetDevicePool(DeviceConfig.Host_Unkown).EraseAll();
            Console.WriteLine(Tensor.DisposedCount);
            Thread.Sleep(3000);
        }
    }
}
