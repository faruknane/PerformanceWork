using PerformanceWork.NCuda;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Runtime.CompilerServices;

namespace PerformanceWork.DeepLearning.Kernels.NvidiaGpu
{
    public unsafe static partial class NvidiaGpuKernels
    {
        /// <summary>
        /// Returns the sum of tensors.
        /// </summary>
        /// <param name="tensors">Tensors to be summed</param>
        /// <returns>The sum of tensors</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor AddFloat32(Tensor a, Tensor b, float cofa = 1, float cofb = 1, float cofadd = 0)
        {
            Tensor res = new Tensor(a.Shape.Clone(), b.Config);
            AddFloat32_Result(res, a, b, cofa, cofb, cofadd);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32_Result(Tensor res, Tensor a, Tensor b, float cofa = 1, float cofb = 1, float cofadd = 0)
        {
            CudaManagement.SetDevice(res.Config.Device.ID);
            CudaKernels.AddFloat32((float*)res.Array, (float*)a.Array, (float*)b.Array, a.Shape.TotalSize, b.Shape.TotalSize, cofa, cofb, cofadd);
        }

    }
}
