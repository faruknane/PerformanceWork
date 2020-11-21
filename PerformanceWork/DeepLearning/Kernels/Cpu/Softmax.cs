using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.DeepLearning.Kernels.Cpu
{
    public unsafe partial class CpuKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SoftmaxFloat(Tensor v)
        {
            Tensor sm = new Tensor(v.Shape.Clone(), DeviceConfig.Host_Float);
            VectorizationFloat.Softmax((float*)v.Array, (float*)sm.Array, v.Shape[v.Shape.N - 1], v.Shape.TotalSize);
            return sm;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SoftmaxFloat_GetGradient_0(Tensor s, Tensor sm)
        {
            Tensor combined = Tensor.Clone(s);
            
            int groupsize = sm.Shape[sm.Shape.N - 1];

            for (int start = 0; start < combined.Shape.TotalSize; start += groupsize)
            {
                float averageK = VectorizationFloat.SumOfProduction((float*)s.Array + start, (float*)sm.Array + start, groupsize);
                VectorizationFloat.ElementWiseAddAVX((float*)combined.Array + start, -averageK, (float*)combined.Array + start, groupsize);
            }

            VectorizationFloat.ElementWiseMultiplyAVX((float*)combined.Array, (float*)sm.Array, (float*)combined.Array, combined.Shape.TotalSize);

            return combined;
        }

    }
}
