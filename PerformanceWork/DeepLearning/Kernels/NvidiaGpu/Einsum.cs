using PerformanceWork.NCuda;
using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Runtime.CompilerServices;
using static PerformanceWork.NCuda.CudaTypes;

namespace PerformanceWork.DeepLearning.Kernels.NvidiaGpu
{
    public unsafe static partial class NvidiaGpuKernels
    {
        //todo write the function definition
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Einsum(Tensor D,
            Tensor A, string modeA, Tensor B, string modeB, Tensor C, string modeC, 
            double alpha = 1f, double beta = 0f)
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

            fixed (int* ptrmA = mA,
                ptrmB = mB,
                ptrmC = mC)
            fixed (long* DimensionsA = A.Shape.Dimensions, MultipliedA = A.Shape.Multiplied,
            DimensionsB = B.Shape.Dimensions, MultipliedB = B.Shape.Multiplied,
            DimensionsC = C.Shape.Dimensions, MultipliedC = C.Shape.Multiplied,
            DimensionsD = D.Shape.Dimensions, MultipliedD = D.Shape.Multiplied)
                CudaKernels.Einsum(
                    A.Base.Array, A.Shape.N, ptrmA, DimensionsA, MultipliedA + 1, CudaTypes.GetDataType(A),
                    B.Base.Array, B.Shape.N, ptrmB, DimensionsB, MultipliedB + 1, CudaTypes.GetDataType(B),
                    C.Base.Array, C.Shape.N, ptrmC, DimensionsC, MultipliedC + 1, CudaTypes.GetDataType(C),
                    D.Base.Array, D.Shape.N, ptrmC, DimensionsD, MultipliedD + 1, CudaTypes.GetDataType(D),
                    alpha, beta, CudaTypes.GetComputeType(C));
        }

        //Todo: noktalı kullanım ve Tensor D siz olan function da yaz.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Einsum(Tensor D, string op, 
            Tensor A, Tensor B, Tensor C, double alpha = 1f, double beta = 0f)
        {
            var sp = op.Split("->", StringSplitOptions.None);

            if (sp.Length != 2)
                throw new Exception("Operation is illegal!");

            var sp2 = sp[0].Split(',', StringSplitOptions.None);

            if (sp2.Length != 2)
                throw new Exception("Operation is illegal!");

            string sA = sp2[0];
            string sB = sp2[1];
            string sC = sp[1];
            Einsum(D, A, sA, B, sB, C, sC, alpha, beta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MatrixMultiplyFloat32(Tensor res, Tensor a, Tensor b, float cofmul = 1, float cofadd = 0)
        {

        }

    }
}
