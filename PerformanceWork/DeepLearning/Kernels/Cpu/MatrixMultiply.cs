using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.DeepLearning.Kernels.Cpu
{
    /// <summary>
    /// TODO: should support more than 2 dimensions.
    /// </summary>
    public unsafe partial class CpuKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiplyFloat32_GetGradient_0(Tensor s, Tensor B, Shape thisShape, Shape term0, Shape term1)
        {
            var combinedleft = new Tensor(term0.Clone(), TensorConfig.Host_Float32);
            MatrixMultiplyFloat32_GetGradient_0(combinedleft, s, B, thisShape, term0, term1);
            return combinedleft;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MatrixMultiplyFloat32_GetGradient_0(Tensor combinedleft, Tensor s, Tensor B, Shape thisShape, Shape term0, Shape term1)
        {
            float* ptr_left = (float*)combinedleft.Base.Array, ptr_s = (float*)s.Base.Array, ptr_b = (float*)B.Base.Array;
            VectorizationFloat.TransposeBandMatrixMultiply(ptr_s, (int)thisShape[0], (int)thisShape[1], ptr_b, (int)B.Shape[0], (int)B.Shape[1], ptr_left);
            //Derivative of A = s*Transpose(B)
            //A -> m,k
            //B -> k,n
            //s -> m,n
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiplyFloat32_GetGradient_1(Tensor s, Tensor A, Shape thisShape, Shape term0, Shape term1)
        {
            var combinedright = new Tensor(term1.Clone(), TensorConfig.Host_Float32);
            MatrixMultiplyFloat32_GetGradient_1(combinedright, s, A, thisShape, term0, term1);
            return combinedright;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MatrixMultiplyFloat32_GetGradient_1(Tensor combinedright, Tensor s, Tensor A, Shape thisShape, Shape term0, Shape term1)
        {
            float* ptr_right = (float*)combinedright.Base.Array, ptr_a = (float*)A.Base.Array, ptr_s = (float*)s.Base.Array;
            VectorizationFloat.TransposeAandMatrixMultiply(ptr_a, A.Shape[0], A.Shape[1], ptr_s, thisShape[0], thisShape[1], ptr_right);
            //Derivative of B = Transpose(A)*s
            //A -> m,k
            //B -> k,n
            //s -> m,n
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiplyFloat32(Tensor a, Tensor b)
        {
            Shape sc = new Shape(a.Shape[0], b.Shape[1]);
            Tensor res = new Tensor(sc, a.Config);
            MatrixMultiplyFloat32(res, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MatrixMultiplyFloat32(Tensor res, Tensor a, Tensor b)
        {
            VectorizationFloat.MatrixMultiply(a, b, res);
        }

    }
}
