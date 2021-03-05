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
        public static Tensor SigmoidFloat32(Tensor v)
        {
            Tensor res = new Tensor(v.Shape.Clone(), TensorConfig.Host_Float32);
            SigmoidFloat32(res, v);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SigmoidFloat32(Tensor res, Tensor v)
        {
            VectorizationFloat.Sigmoid((float*)v.Array, (float*)res.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SigmoidFloat32_GetGradient_0(Tensor s, Tensor sigmo)
        {
            Tensor combined = new Tensor(s.Shape.Clone(), TensorConfig.Host_Float32);
            SigmoidFloat32_GetGradient_0(combined, s, sigmo);
            return combined;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SigmoidFloat32_GetGradient_0(Tensor combined, Tensor s, Tensor sigmo)
        {
            VectorizationFloat.ElementWise_A_MultipliedBy_1_Minus_A_MultipliedByB((float*)sigmo.Array, (float*)s.Array, (float*)combined.Array, sigmo.Shape.TotalSize);
        }

    }
}
