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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor ReluFloat32(Tensor v)
        {
            Tensor res = new Tensor(v.Shape.Clone(), TensorConfig.Host_Float32);
            ReluFloat32(res, v);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void ReluFloat32(Tensor res, Tensor v)
        {
            VectorizationFloat.FilterNegativeNumbers((float*)v.Base.Array, (float*)res.Base.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor ReluFloat32_GetGradient_0(Tensor gradient, Tensor v)
        {
            Tensor combined = new Tensor(gradient.Shape.Clone(), TensorConfig.Host_Float32);
            ReluFloat32_GetGradient_0(combined, gradient, v);
            return combined;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void ReluFloat32_GetGradient_0(Tensor combined, Tensor gradient, Tensor v)
        {
            VectorizationFloat.ReluFloatGradientCalculation((float*)gradient.Base.Array, (float*)v.Base.Array, (float*)combined.Base.Array, combined.Shape.TotalSize);
        }

    }
}
