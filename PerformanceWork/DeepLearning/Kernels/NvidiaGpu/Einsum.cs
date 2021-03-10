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
        public static void Einsum(Tensor D, 
            Tensor A, string modeA, Tensor B, string modeB, Tensor C, string modeC, float alpha = 1f, float beta = 0f)
        {
            int[] mA = new int[modeA.Length];
            int[] mB = new int[modeB.Length];
            int[] mC = new int[modeC.Length];

            for (int i = 0; i < mA.Length; i++)
                mA[i] = modeA[i];

            for (int i = 0; i < mB.Length; i++)
                mB[i] = modeB[i];

            for (int i = 0; i < mC.Length; i++)
                mC[i] = modeC[i];

            fixed(int* ptrmA = mA,
                ptrmB = mB,
                ptrmC = mC)
                fixed(long* DimensionsA = A.Shape.Dimensions,
                DimensionsB = B.Shape.Dimensions,
                DimensionsC = C.Shape.Dimensions)
                    CudaKernels.Einsum(
                        A.Base.Array, modeA.Length, ptrmA, DimensionsA, CudaDataType.CUDA_R_32F,
                        B.Base.Array, modeB.Length, ptrmB, DimensionsB, CudaDataType.CUDA_R_32F,
                        C.Base.Array, modeC.Length, ptrmC, DimensionsC, CudaDataType.CUDA_R_32F,
                        D.Base.Array, modeC.Length, ptrmC, DimensionsC, CudaDataType.CUDA_R_32F,
                        &alpha, &beta, CutensorComputeType.CUTENSOR_COMPUTE_32F);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MatrixMultiplyFloat32(Tensor res, Tensor a, Tensor b, float cofmul = 1, float cofadd = 0)
        {

        }

    }
}
