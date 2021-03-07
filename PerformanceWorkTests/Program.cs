using PerformanceWork;
using PerformanceWork.DeepLearning.Kernels.Cpu;
using PerformanceWork.DeepLearning.Kernels.NvidiaGpu;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Pool;
using PerformanceWork.OptimizedNumerics.Tensors;
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
        public static void f()
        {
            for (int i = 0; i < 100; i++)
            {
                //4mb
                Tensor a = new Tensor(new Shape(1024 * 1024), TensorConfig.Host_Float32);
                a.Dispose();
            }
            
        }

        public static void f2()
        {
            var f = new float[10] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            Tensor b = f.ToDisposedTensor(new Shape(10), NumberType.Float32);
        }


        static unsafe void Main(string[] args)
        {
            f2();
            Console.WriteLine(Tensor.DisposedCount);
            GC.Collect();
            Thread.Sleep(100);
            Console.WriteLine(Tensor.DisposedCount);
            GC.Collect();
            Thread.Sleep(100);
            Console.WriteLine(Tensor.DisposedCount);
            TensorPool.GetDevicePool(Device.Host).EraseAll();
            TensorPool.GetDevicePool(Device.Nvidia(0)).EraseAll();
            Console.WriteLine(Tensor.DisposedCount);
            Thread.Sleep(3000);
        }
    }
}
