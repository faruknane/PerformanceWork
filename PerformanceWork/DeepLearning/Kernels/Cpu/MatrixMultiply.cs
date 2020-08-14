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

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiplyFloat_GetGradient_0(Tensor s, Tensor B, Shape thisShape, Shape term0, Shape term1)
        {
            var combinedleft = new Tensor(term0.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            float* ptr_left = (float*)combinedleft.Array, ptr_s = (float*)s.Array, ptr_b = (float*)B.Array;
            VectorizationFloat.TransposeBandMatrixMultiply(ptr_s, thisShape[0], thisShape[1], ptr_b, B.Shape[0], B.Shape[1], ptr_left);
            return combinedleft;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiplyFloat_GetGradient_1(Tensor s, Tensor A, Shape thisShape, Shape term0, Shape term1)
        {
            var combinedright = new Tensor(term1.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            float* ptr_right = (float*)combinedright.Array, ptr_a = (float*)A.Array, ptr_s = (float*)s.Array;
            VectorizationFloat.TransposeAandMatrixMultiply(ptr_a, A.Shape[0], A.Shape[1], ptr_s, thisShape[0], thisShape[1], ptr_right);
            return combinedright;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiplyFloat(Tensor a, Tensor b)
        {
            Shape sc = new Shape((a.Shape[0], b.Shape[1]));
            Tensor c = new Tensor(sc, a.Type, a.Device);
            VectorizationFloat.MatrixMultiply(a, b, c);
            return c;
        }

    }
}
