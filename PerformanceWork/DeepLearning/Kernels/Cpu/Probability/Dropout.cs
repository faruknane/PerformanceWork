using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.DeepLearning.Kernels.Cpu
{
    public unsafe static partial class CpuKernels
    {
        public unsafe static partial class Probability
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static void DropoutFloat(Tensor s, float p)
            {
                Random r = new Random();
                float* ptr = (float*)s.Base.Array;
                for (int i = 0; i < s.Shape.TotalSize; i++)
                    if (r.NextDouble() > p)
                        ptr[i] = 1;
                    else
                        ptr[i] = 0;
            }
        }
    }
  
}
