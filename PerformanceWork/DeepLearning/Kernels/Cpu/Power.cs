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
        /// <summary>
        /// Calculates the gradient of the power kernel.
        /// </summary>
        /// <param name="s">The gradient tensor</param>
        /// <param name="res"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Power2Float_GetGradient_0(Tensor s, Tensor res)
        {
            Tensor combined = new Tensor(s.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            float* ptr_combined = (float*)combined.Array;
            float* ptr_s = (float*)s.Array;
            VectorizationFloat.ElementWise_A_MultipliedBy_B_MultipliedBy_C((float*)res.Array, ptr_s, 2, ptr_combined, res.Shape.TotalSize);
            return combined;
        }

        /// <summary>
        /// Calculates the power
        /// </summary>
        /// <param name="res"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Power2Float(Tensor res)
        {
            Tensor m = new Tensor(res.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWiseSquareAVX((float*)res.Array, (float*)m.Array, m.Shape.TotalSize);
            return m;
        }
    }
}
