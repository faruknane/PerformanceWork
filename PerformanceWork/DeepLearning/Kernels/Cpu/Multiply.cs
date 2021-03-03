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
        public static Tensor MultiplyFloat(Tensor a, Tensor b)
        {
            Tensor res = new Tensor(a.Shape.Clone(), DeviceConfig.Host_Float32);
            VectorizationFloat.ElementWiseMultiplyAVX((float*)a.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat(Tensor res, Tensor a, Tensor b)
        {
            VectorizationFloat.ElementWiseMultiplyAVX((float*)a.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MultiplyFloat_GetGradient_0(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(s.Shape.Clone(), DeviceConfig.Host_Float32);
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat_GetGradient_0(Tensor res, Tensor s, Tensor a, Tensor b)
        {
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MultiplyFloat_GetGradient_1(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(s.Shape.Clone(), DeviceConfig.Host_Float32);
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)a.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat_GetGradient_1(Tensor res, Tensor s, Tensor a, Tensor b)
        {
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)a.Array, (float*)res.Array, res.Shape.TotalSize);
        }
    }
}