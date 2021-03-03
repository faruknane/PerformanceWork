using PerformanceWork.OptimizedNumerics;
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
        public static Tensor ReluFloat(Tensor v)
        {
            Tensor res = new Tensor(v.Shape.Clone(), DeviceConfig.Host_Float32);
            VectorizationFloat.FilterNegativeNumbers((float*)v.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void ReluFloat(Tensor res, Tensor v)
        {
            VectorizationFloat.FilterNegativeNumbers((float*)v.Array, (float*)res.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor ReluFloat_GetGradient_0(Tensor gradient, Tensor v)
        {
            Tensor combined = new Tensor(gradient.Shape.Clone(), DeviceConfig.Host_Float32);
            VectorizationFloat.ReluFloatGradientCalculation((float*)gradient.Array, (float*)v.Array, (float*)combined.Array, combined.Shape.TotalSize);
            return combined;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void ReluFloat_GetGradient_0(Tensor combined, Tensor gradient, Tensor v)
        {
            VectorizationFloat.ReluFloatGradientCalculation((float*)gradient.Array, (float*)v.Array, (float*)combined.Array, combined.Shape.TotalSize);
        }

    }
}
