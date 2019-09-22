using System;

namespace PerformanceWork.OptimizedNumerics
{
    public partial class Vectorization
    {
        public unsafe struct PointerCarrier
        {
            public float* ptr;
        }

        public static unsafe void MatrixMultiply(Matrix a, Matrix b, Matrix c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.NoTrans, a.D1, b.D2, b.D1, 1.0f, a.GetPointer(), b.D1, b.GetPointer(), b.D2, 0.0f, c.GetPointer(), b.D2);
        }

        public static unsafe void MatrixMultiply(float* a, int ad1, int ad2, float* b, int bd1, int bd2, float* c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.NoTrans, ad1, bd2, bd1, 1.0f, a, bd1, b, bd2, 0.0f, c, bd2);
        }

        public static unsafe void Exponential(float* inp, float* outp, int length)
        {
            MKL.vmsExp(length, inp, outp, 0x00000003);
        }

        public static unsafe void Sigmoid(float* inp, float* outp, int length)
        {
            Vectorization.ElementWiseMultiplyAVX(inp, -1, outp, length);
            MKL.vmsExp(length, outp, outp, 0x00000003);
            Vectorization.ElementWiseAddAVX(outp, 1, outp, length);
            Vectorization.ElementWiseDivideAVX(1, outp, outp, length);
        }
        public static unsafe void TransposeBandMatrixMultiply(float* a, int ad1, int ad2, float* b, int bd1, int bd2, float* c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.NoTrans, MKL.TRANSPOSE.Trans, ad1, bd1, bd2, 1.0f, a, bd2, b, bd2, 0.0f, c, bd1);
        }
        public static unsafe void TransposeAandMatrixMultiply(float* a, int ad1, int ad2, float* b, int bd1, int bd2, float* c)
        {
            MKL.cblas_sgemm(MKL.ORDER.RowMajor, MKL.TRANSPOSE.Trans, MKL.TRANSPOSE.NoTrans, ad2, bd2, bd1, 1.0f, a, ad2, b, bd2, 0.0f, c, bd2);
        }
        //public static unsafe void Tranpose(Matrix a)
        //{
        //    MKL.mkl_simatcopy('r', 't', a.D1, a.D2, 1.0f, a.Array, a.D1, a.D2);
        //}

        //public static unsafe void Copy(float* a, int ad1, int ad2, int xa, int ya, int lxa, int lya, float* b, int bd1, int bd2, int xb, int xy, int lxb, int lyb)
        //{
        //    MKL.pslacpy('o', lxa, lya, a,)
        //}

    }
}
