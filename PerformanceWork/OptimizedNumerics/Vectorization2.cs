using PerformanceWork.OptimizedNumerics.Tensors;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics
{
    public partial class VectorizationFloat
    {
        public static int Mode = 0x00000003;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatrixMultiply(Tensor a, Tensor b, Tensor c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.NoTrans, a.Shape[0], b.Shape[1], b.Shape[0], 1.0f, (float*)a.Array, b.Shape[0], (float*)b.Array, b.Shape[1], 0.0f, (float*)c.Array, b.Shape[1]);
        }
     
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatrixMultiply(float* a, long ad1, long ad2, float* b, long bd1, long bd2, float* c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.NoTrans, ad1, bd2, bd1, 1.0f, a, bd1, b, bd2, 0.0f, c, bd2);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Exponential(float* inp, float* outp, long length)
        {
            MKL.vmsExp(length, inp, outp, Mode);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Sigmoid(float* inp, float* outp, long length)
        {
            VectorizationFloat.ElementWiseMultiplyAVX(inp, -1, outp, length);
            MKL.vmsExp(length, outp, outp, Mode);
            VectorizationFloat.ElementWise_A_DividedBy_B_Plus_Vector(1, 1, outp, outp, length);
        }

        public static unsafe void ElementWise_A_DividedBy_B_Plus_Vector(float a, float b, float* vector, float* ptr_res, long length)
        {
            float* ptr_vala = &a;
            float* ptr_valb = &b;
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&vector[i]);
                Vector256<float> va = Avx2.BroadcastScalarToVector256(ptr_vala);
                Vector256<float> vb = Avx2.BroadcastScalarToVector256(ptr_valb);
                Vector256<float> res = Avx2.Divide(va, Avx2.Add(vb, v1));
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = a / (b + vector[i]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void TransposeBandMatrixMultiply(float* a, long ad1, long ad2, float* b, long bd1, long bd2, float* c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.Trans, ad1, bd1, bd2, 1.0f, a, bd2, b, bd2, 0.0f, c, bd1);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void TransposeAandMatrixMultiply(float* a, long ad1, long ad2, float* b, long bd1, long bd2, float* c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.Trans, MKL.TRANSPOSE.NoTrans, ad2, bd2, bd1, 1.0f, a, ad2, b, bd2, 0.0f, c, bd2);
        }

        public static unsafe void ElementWise_A_MultipliedBy_B_MultipliedBy_C(float* ptr_a, float* ptr_b, float c, float* ptr_res, long length)
        {
            float* ptr_c = &c;
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> vb = Avx2.LoadVector256(&ptr_b[i]);
                Vector256<float> vc = Avx2.BroadcastScalarToVector256(ptr_c);
                Vector256<float> res = Avx2.Multiply(va, Avx2.Multiply(vb, vc));
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_a[i] * ptr_b[i] * c;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWise_A_MultipliedBy_1_Minus_A(float* ptr_a, float* ptr_res, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            float valone = 1;
            Vector256<float> one = Avx2.BroadcastScalarToVector256(&valone);
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> res = Avx2.Multiply(va, Avx2.Subtract(one, va));
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_a[i] * (1 - ptr_a[i]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWise_A_MultipliedBy_1_Minus_A_MultipliedByB(float* ptr_a, float* ptr_b, float* ptr_res, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            float valone = 1;
            Vector256<float> one = Avx2.BroadcastScalarToVector256(&valone);
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> vb = Avx2.LoadVector256(&ptr_b[i]);
                Vector256<float> res = Avx2.Multiply(va, Avx2.Subtract(one, va));
                res = Avx2.Multiply(res, vb);
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_a[i] * (1 - ptr_a[i]) * ptr_b[i];
            }
        }

        public static unsafe void Softmax(float* source, float* res, long groupsize, long length)
        {
            for (long j = 0; j < length; j += groupsize)
            {
                float maxelement = FindMaxElement(source + j, groupsize);
                ElementWiseSubtractAVX(source + j, maxelement, res + j, groupsize);
                Exponential(res + j, res + j, groupsize);
                float sum = SumOfVector(res + j, groupsize);
                ElementWiseDivideAVX(res + j, sum, res + j, groupsize);
            }
        }

        private static unsafe float FindMaxElement(float* source, long length)
        {
            float maxel = float.MinValue;
            for (long i = 0; i < length; i++)
            {
                float val = source[i];
                if (maxel < val)
                    maxel = val;
            }
            return maxel;
        }

        public static unsafe void Print(float* a, int l)
        {
            for (int i = 0; i < l; i++)
                Console.Write(a[i] + (i == l - 1 ? "" : ", "));
            Console.WriteLine();
        }

    }
}
