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
        public static Tensor SubtractFloat32(Tensor t1, Tensor t2)
        {
            Tensor res = new Tensor(t1.Shape.Clone(), TensorConfig.Host_Float32);
            SubtractFloat32(res, t1, t2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SubtractFloat32(Tensor res, Tensor t1, Tensor t2)
        {
            VectorizationFloat.ElementWiseSubtractAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SubtractFloat32_GetGradient_1(Tensor m, Tensor t1, Tensor t2)
        {
            Tensor res = new Tensor(m.Shape.Clone(), TensorConfig.Host_Float32);
            SubtractFloat32_GetGradient_1(res, m, t1, t2);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SubtractFloat32_GetGradient_1(Tensor res, Tensor m, Tensor t1, Tensor t2)
        {
            VectorizationFloat.MakeNegativeAVX((float*)m.Array, (float*)res.Array, m.Shape.TotalSize);
        }


    }
}
