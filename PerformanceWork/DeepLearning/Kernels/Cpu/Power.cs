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
        public static Tensor Power2Float32_GetGradient_0(Tensor s, Tensor res)
        {
            Tensor combined = new Tensor(s.Shape.Clone(), TensorConfig.Host_Float32);
            Power2Float32_GetGradient_0(combined, s, res);
            return combined;
        }

        /// <summary>
        /// Calculates the gradient of the power kernel.
        /// </summary>
        /// <param name="s">The gradient tensor</param>
        /// <param name="res"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Power2Float32_GetGradient_0(Tensor combined, Tensor s, Tensor res)
        {
            float* ptr_combined = (float*)combined.Array;
            float* ptr_s = (float*)s.Array;
            VectorizationFloat.ElementWise_A_MultipliedBy_B_MultipliedBy_C((float*)res.Array, ptr_s, 2, ptr_combined, res.Shape.TotalSize);
        }

        /// <summary>
        /// Calculates the power
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Power2Float32(Tensor input)
        {
            Tensor res = new Tensor(input.Shape.Clone(), TensorConfig.Host_Float32);
            Power2Float32(res, input);
            return res;
        }

        /// <summary>
        /// Calculates the power
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Power2Float32(Tensor res, Tensor input)
        {
            VectorizationFloat.ElementWiseSquareAVX((float*)input.Array, (float*)res.Array, res.Shape.TotalSize);
        }
    }
}
