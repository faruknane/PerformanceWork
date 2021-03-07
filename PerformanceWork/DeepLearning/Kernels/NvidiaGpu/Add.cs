using PerformanceWork.NCuda;
using PerformanceWork.OptimizedNumerics;
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
        public static Tensor AddFloat32(params Tensor[] tensors)
        {
            for (int i = 0; i < tensors.Length - 1; i++)
                if (tensors[i].Config != tensors[i + 1].Config || tensors[i].Shape.TotalSize != tensors[i+1].Shape.TotalSize)
                    throw new Exception("Tensors are not suitable!");

            Tensor res = new Tensor(tensors[0].Shape.Clone(), tensors[0].Config);
            AddFloat32_Result(res, tensors);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32_Result(Tensor res, params Tensor[] tensors)
        {
            CudaManagement.SetDevice(res.Config.Device.ID);
            if (tensors.Length == 2)
                CudaKernels.AddFloat32((float*)res.Array, (float*)tensors[0].Array, (float*)tensors[1].Array, (int)res.Shape.TotalSize);
            else
                throw new Exception("More than 2 tensors are not supported for the operation!");
        }

    }
}
