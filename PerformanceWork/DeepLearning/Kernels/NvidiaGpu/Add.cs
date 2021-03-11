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
        /// Assumes that Tensor A is larger than or equal to Tensor B. 
        /// </summary>
        /// <param name="a">Tensors to be summed.</param>
        /// <param name="b">Tensors to be summed.</param>
        /// <returns>Returns the sum of A and B tensors.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor AddFloat32(Tensor a, Tensor b, float cofa = 1, float cofb = 1, float cofadd = 0)
        {
            Tensor res = new Tensor(a.Shape.Clone(), b.Config);
            AddFloat32(res, a, b, cofa, cofb, cofadd);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32(Tensor res, Tensor a, Tensor b, float cofa = 1, float cofb = 1, float cofadd = 0)
        {
            CudaManagement.SetDevice(res.Config.Device.ID);
            CudaKernels.AddFloat32((float*)res.Base.Array, (float*)a.Base.Array, (float*)b.Base.Array, a.Shape.TotalSize, b.Shape.TotalSize, cofa, cofb, cofadd);
        }

    }
}
