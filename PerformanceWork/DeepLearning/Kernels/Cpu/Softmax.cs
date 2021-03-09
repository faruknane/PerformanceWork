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
    public unsafe partial class CpuKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SoftmaxFloat32(Tensor v)
        {
            Tensor res = new Tensor(v.Shape.Clone(), TensorConfig.Host_Float32);
            SoftmaxFloat32(res, v);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SoftmaxFloat32(Tensor res, Tensor v)
        {
            VectorizationFloat.Softmax((float*)v.Base.Array, (float*)res.Base.Array, v.Shape[v.Shape.N - 1], v.Shape.TotalSize);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SoftmaxFloat32_GetGradient_0(Tensor s, Tensor sm)
        {
            Tensor combined = Tensor.Clone(s);
            
            long groupsize = sm.Shape[sm.Shape.N - 1];

            for (long start = 0; start < combined.Shape.TotalSize; start += groupsize)
            {
                float averageK = VectorizationFloat.SumOfProduction((float*)s.Base.Array + start, (float*)sm.Base.Array + start, groupsize);
                VectorizationFloat.ElementWiseAddAVX((float*)combined.Base.Array + start, -averageK, (float*)combined.Base.Array + start, groupsize);
            }

            VectorizationFloat.ElementWiseMultiplyAVX((float*)combined.Base.Array, (float*)sm.Base.Array, (float*)combined.Base.Array, combined.Shape.TotalSize);

            return combined;
        }

    }
}
