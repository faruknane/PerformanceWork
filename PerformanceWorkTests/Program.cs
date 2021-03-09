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
            TensorBase t = new TensorBase(100, TensorConfig.Host_Float32);

            for (int i = 0; i < 100; i++)
            {
                //4mb
                Tensor a = new Tensor(new Shape(100), t);
            }
            
        }

        static unsafe void Main(string[] args)
        {
            f();
            Console.WriteLine(TensorBase.DisposedCount);
            GC.Collect();
            Thread.Sleep(100);
            Console.WriteLine(TensorBase.DisposedCount);
            GC.Collect();
            Thread.Sleep(100);
            Console.WriteLine(TensorBase.DisposedCount);
            TensorPool.GetDevicePool(Device.Host).EraseAll();
            TensorPool.GetDevicePool(Device.Nvidia(0)).EraseAll();
            Console.WriteLine(TensorBase.DisposedCount);
            Thread.Sleep(3000);
        }
    }
}
