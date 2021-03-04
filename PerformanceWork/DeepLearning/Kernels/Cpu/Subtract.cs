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
        public static Tensor SubtractFloat(Tensor t1, Tensor t2)
        {
            Tensor res = new Tensor(t1.Shape.Clone(), TensorConfig.Host_Float32);
            VectorizationFloat.ElementWiseSubtractAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SubtractFloat_GetGradient_0(Tensor m, Tensor t1, Tensor t2)
        {
            return m;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SubtractFloat_GetGradient_1(Tensor m, Tensor t1, Tensor t2)
        {
            Tensor res = new Tensor(m.Shape.Clone(), TensorConfig.Host_Float32);
            VectorizationFloat.MakeNegativeAVX((float*)m.Array, (float*)res.Array, m.Shape.TotalSize);
            return res;
        }


    }
}
