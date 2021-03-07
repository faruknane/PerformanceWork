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
        public static Tensor MultiplyFloat32(Tensor a, Tensor b)
        {
            Tensor res = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
            MultiplyFloat32(res, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat32(Tensor res, Tensor a, Tensor b)
        {
            VectorizationFloat.ElementWiseMultiplyAVX((float*)a.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MultiplyFloat32_GetGradient_0(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(s.Shape.Clone(), TensorConfig.Host_Float32);
            MultiplyFloat32_GetGradient_0(res, s, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat32_GetGradient_0(Tensor res, Tensor s, Tensor a, Tensor b)
        {
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MultiplyFloat32_GetGradient_1(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(s.Shape.Clone(), TensorConfig.Host_Float32);
            MultiplyFloat32_GetGradient_1(res, s, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat32_GetGradient_1(Tensor res, Tensor s, Tensor a, Tensor b)
        {
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)a.Array, (float*)res.Array, res.Shape.TotalSize);
        }
    }
}