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
            Tensor a = new Tensor(new Shape(450000000), TensorConfig.NvidiaGPU_Float32);
            Tensor b = new Tensor(new Shape(400000000), TensorConfig.NvidiaGPU_Float32);
            NvidiaGpuKernels.AddFloat32(a, a, b);
            Thread.Sleep(3000);

            a.Dispose();
            b.Dispose();
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
