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
        public static void f()
        {
            for (int i = 0; i < 100; i++)
            {
                //4mb
                Tensor a = new Tensor(new Shape(1024 * 1024), TensorConfig.NvidiaGPU_Unkown);
                a.Dispose();
            }
        }

        static unsafe void Main(string[] args)
        {
            int size = 100000000;
            var f = new float[size];
            Tensor a = f.ToDisposedTensor(new Shape(size), NumberType.Float32);
            Tensor b = Tensor.CopyTo(a, Device.Nvidia(0));
            Thread.Sleep(5000);
            //Console.WriteLine(b);
            b.Dispose();
            Thread.Sleep(5000);
            return;
            //f();
            //Console.WriteLine(Tensor.DisposedCount);
            //TensorPool.GetDevicePool(Device.Host).EraseAll();
            //TensorPool.GetDevicePool(Device.Nvidia(0)).EraseAll();
            //Console.WriteLine(Tensor.DisposedCount);
            //Thread.Sleep(3000);
        }
    }
}
