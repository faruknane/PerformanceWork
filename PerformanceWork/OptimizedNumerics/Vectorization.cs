using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics
{
    public partial class Vectorization
    {

        #region Working Properly

        /// <summary>
        /// result = left + right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        /// <param name="length"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void AddAVX(ref float[] left, ref float[] right, ref float[] result, int length)
        {
            fixed (float* ptr_a = left, ptr_b = right, ptr_res = result)
            {
                for (long i = 0; i < length; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    Vector256<float> res = Avx2.Add(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
            }
        }
        /// <summary>
        /// result =  left Dot right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        /// <param name="length"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProductAVX(ref float[] left, ref float[] right, int length)
        {
            //editliyorum
            fixed (float* ptr_a = left, ptr_b = right)
            {
                long remain = length % Vector256<float>.Count;

                Vector256<float> sum = new Vector256<float>();
                long i = 0;
                while (i < length - remain)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    sum = Fma.MultiplyAdd(v2, v1, sum);
                    i += Vector256<float>.Count;
                }

                float result = 0;
                sum = Fma.HorizontalAdd(sum, sum);
                sum = Fma.HorizontalAdd(sum, sum);
                result = sum.GetElement(0) + sum.GetElement(4);
                float remainingsum = 0;
                for (i = length - remain; i < length; i++)
                {
                    remainingsum += (ptr_a[i]) * (ptr_b[i]);
                }
                result += remainingsum;
                return result;
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProductAVX(float* ptr_a, float* ptr_b, long length)
        {
            long remain = length % Vector256<float>.Count;

            Vector256<float> sum = new Vector256<float>();
            long i = 0;
            while (i < length - remain)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                sum = Fma.MultiplyAdd(v2, v1, sum);
                i += Vector256<float>.Count;
            }

            float result = 0;
            sum = Fma.HorizontalAdd(sum, sum);
            sum = Fma.HorizontalAdd(sum, sum);
            result = sum.GetElement(0) + sum.GetElement(4);
            float remainingsum = 0;
            for (i = length - remain; i < length; i++)
            {
                remainingsum += (ptr_a[i]) * (ptr_b[i]);
            }
            result += remainingsum;
            return result;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProductAVXParallel(float* ptr_a, float* ptr_b, long length)
        {
            int degree = 0;
            int num;
            ThreadPool.GetMinThreads(out degree, out num);
            PointerCarrier carrierA = new PointerCarrier();
            PointerCarrier carrierB = new PointerCarrier();
            carrierA.ptr = ptr_a;
            carrierB.ptr = ptr_b;

            long[] blocks = new long[degree];

            for (int i = 0; i < degree; i++)
                blocks[i] = length / degree;

            blocks[degree-1] += length % degree;

            long[] starts = new long[degree];
            long sum = 0;
            for (int i = 0; i < degree; i++)
            {
                starts[i] = sum;
                sum += blocks[i];
            }

            float result = 0;
            object locker = new object();
            Parallel.For(0, degree, new ParallelOptions() { MaxDegreeOfParallelism = degree} ,(long threadno) =>
            {
                float ownres = DotProductAVX(carrierA.ptr + starts[threadno], carrierB.ptr + starts[threadno], blocks[threadno]);
                lock (locker)
                {
                    result += ownres;
                }
            });
            return result;
        }
        /// <summary>
        /// Tranpose of a 8x8 Matrix
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        /// <param name="length"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MatrixTranspose8x8(ref float[] matrix, ref float[] result)
        {
            fixed (float* ptr_a = matrix, ptr_res = result)
            {
                Vector256<float> x0 = Avx.LoadVector256(&ptr_a[0]);
                Vector256<float> x1 = Avx.LoadVector256(&ptr_a[8]);
                Vector256<float> x2 = Avx.LoadVector256(&ptr_a[16]);
                Vector256<float> x3 = Avx.LoadVector256(&ptr_a[24]);
                Vector256<float> x4 = Avx.LoadVector256(&ptr_a[32]);
                Vector256<float> x5 = Avx.LoadVector256(&ptr_a[40]);
                Vector256<float> x6 = Avx.LoadVector256(&ptr_a[48]);
                Vector256<float> x7 = Avx.LoadVector256(&ptr_a[56]);

                Vector256<float> y0 = Avx.Shuffle(x0, x1, 0b_0100_0100);
                Vector256<float> y1 = Avx.Shuffle(x0, x1, 0b_1110_1110);
                Vector256<float> y2 = Avx.Shuffle(x2, x3, 0b_0100_0100);
                Vector256<float> y3 = Avx.Shuffle(x2, x3, 0b_1110_1110);
                Vector256<float> y4 = Avx.Shuffle(x4, x5, 0b_0100_0100);
                Vector256<float> y5 = Avx.Shuffle(x4, x5, 0b_1110_1110);
                Vector256<float> y6 = Avx.Shuffle(x6, x7, 0b_0100_0100);
                Vector256<float> y7 = Avx.Shuffle(x6, x7, 0b_1110_1110);

                x0 = Avx.Shuffle(y0, y2, 0b_1000_1000);
                x1 = Avx.Shuffle(y0, y2, 0b_1101_1101);
                x2 = Avx.Shuffle(y1, y3, 0b_1000_1000);
                x3 = Avx.Shuffle(y1, y3, 0b_1101_1101);
                x4 = Avx.Shuffle(y4, y6, 0b_1000_1000);
                x5 = Avx.Shuffle(y4, y6, 0b_1101_1101);
                x6 = Avx.Shuffle(y5, y7, 0b_1000_1000);
                x7 = Avx.Shuffle(y5, y7, 0b_1101_1101);

                y0 = Avx.Permute2x128(x0, x4, 0b_0010_0100);
                y4 = Avx.Permute2x128(x0, x4, 0b_0011_0001);
                y1 = Avx.Permute2x128(x1, x5, 0b_0010_0100);
                y5 = Avx.Permute2x128(x1, x5, 0b_0011_0001);
                y2 = Avx.Permute2x128(x2, x6, 0b_0010_0100);
                y6 = Avx.Permute2x128(x2, x6, 0b_0011_0001);
                y3 = Avx.Permute2x128(x3, x7, 0b_0010_0100);
                y7 = Avx.Permute2x128(x3, x7, 0b_0011_0001);


                Avx.Store(&ptr_res[0], y0);
                Avx.Store(&ptr_res[8], y1);
                Avx.Store(&ptr_res[16], y2);
                Avx.Store(&ptr_res[24], y3);
                Avx.Store(&ptr_res[32], y4);
                Avx.Store(&ptr_res[40], y5);
                Avx.Store(&ptr_res[48], y6);
                Avx.Store(&ptr_res[56], y7);
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MultiplyAVX(ref float[] arr1, ref float[] arr2, ref float[] result, long length)
        {
            fixed (float* ptr_a = arr1, ptr_b = arr2, ptr_res = result)
            {
                for (int i = 0; i < length; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    Vector256<float> res = Avx2.Multiply(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
            }
        }

        #endregion



        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static void MatrixMultiply(ref Matrix a, ref Matrix b, ref Matrix c)
        {
            long m = a.D1, n = a.D2, p = b.D2;
            Matrix bk = new Matrix(b.D1, b.D2);
            Matrix ak = new Matrix(a.D1, a.D2);
            long increment = 14;
            //for (int k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
            //{
            //    for (long j = 0; j < n; ++j)
            //    {
            //        bk[k / 8, j * 8 + 0] = b[j, k + 0];
            //        bk[k / 8, j * 8 + 1] = b[j, k + 1];
            //        bk[k / 8, j * 8 + 2] = b[j, k + 2];
            //        bk[k / 8, j * 8 + 3] = b[j, k + 3];
            //        bk[k / 8, j * 8 + 4] = b[j, k + 4];
            //        bk[k / 8, j * 8 + 5] = b[j, k + 5];
            //        bk[k / 8, j * 8 + 6] = b[j, k + 6];
            //        bk[k / 8, j * 8 + 7] = b[j, k + 7];
            //    }
            //}
            //i j k
            fixed (float* ptr_a = ak.Array, ptr_b = bk.Array, ptr_c = c.Array)
            {
                #region editing ak,bk array with pointers
                for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
                {
                    long positionyforcreating = (k / 8) * 8 * n;
                    for (long j = 0; j < n; ++j)
                    {
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 0];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 1];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 2];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 3];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 4];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 5];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 6];
                        ++positionyforcreating;
                        ptr_b[positionyforcreating] = b.Array[j * p + k + 7];
                        ++positionyforcreating;
                    }
                }

                for (long fi = 0; fi < m / increment * increment; fi += increment)
                {
                    for (long j = 0; j < n; ++j)
                    {
                        //ptr_a[i0 * n + j]
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 0] = a[fi + 0, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 1] = a[fi + 1, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 2] = a[fi + 2, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 3] = a[fi + 3, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 4] = a[fi + 4, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 5] = a[fi + 5, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 6] = a[fi + 6, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 7] = a[fi + 7, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 8] = a[fi + 8, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 9] = a[fi + 9, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 10] = a[fi + 10, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 11] = a[fi + 11, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 12] = a[fi + 12, j];
                        ptr_a[(fi / increment) * (increment * n) + j * increment + 13] = a[fi + 13, j];
                        //ptr_a[(fi / increment) * (increment * n) + j * increment + 14] = a[fi + 14, j];
                    }
                }
                #endregion
                for (long fi = 0; fi < m / increment * increment; fi += increment)
                {
                    long i0 = fi;
                    long i1 = fi + 1;
                    long i2 = fi + 2;
                    long i3 = fi + 3;
                    long i4 = fi + 4;
                    long i5 = fi + 5;
                    long i6 = fi + 6;
                    long i7 = fi + 7;
                    long i8 = fi + 8;
                    long i9 = fi + 9;
                    long i10 = fi + 10;
                    long i11 = fi + 11;
                    long i12 = fi + 12;
                    long i13 = fi + 13;
                    //long i14 = fi + 14;

                    //we can cancel (/ Vector256<float>.Count * Vector256<float>.Count) below
                    for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
                    {
                        // 4 x 8 result
                        Vector256<float> res0 = new Vector256<float>(); //first row of the result
                        Vector256<float> res1 = new Vector256<float>();
                        Vector256<float> res2 = new Vector256<float>();
                        Vector256<float> res3 = new Vector256<float>();
                        Vector256<float> res4 = new Vector256<float>();
                        Vector256<float> res5 = new Vector256<float>();
                        Vector256<float> res6 = new Vector256<float>();
                        Vector256<float> res7 = new Vector256<float>();
                        Vector256<float> res8 = new Vector256<float>();
                        Vector256<float> res9 = new Vector256<float>();
                        Vector256<float> res10 = new Vector256<float>();
                        Vector256<float> res11 = new Vector256<float>();
                        Vector256<float> res12 = new Vector256<float>();
                        Vector256<float> res13 = new Vector256<float>();
                        //Vector256<float> res14 = new Vector256<float>();

                        long positiony = (k / 8) * 8 * n;
                        long postionx = (fi / increment) * (increment * n);

                        //Avx2.Prefetch0(&ptr_b[positiony]);
                        //Avx2.Prefetch0(&ptr_a[postionx]);


                        for (long j = 0; j < n; ++j)
                        {
                            #region real b
                            //Vector256<float> y = Avx2.LoadVector256(&ptr_b[j * p + k]); 
                            #endregion

                            #region bk
                            Vector256<float> y = Avx2.LoadVector256(&ptr_b[positiony + j*8]);
                            #endregion

                            #region real a
                            //res0 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i0 * n + j]), res0);
                            //res1 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i1 * n + j]), res1);
                            //res2 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i2 * n + j]), res2);
                            //res3 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i3 * n + j]), res3);
                            //res4 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i4 * n + j]), res4);
                            //res5 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i5 * n + j]), res5);
                            //res6 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i6 * n + j]), res6);
                            //res7 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i7 * n + j]), res7);
                            //res8 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i8 * n + j]), res8);
                            //res9 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i9 * n + j]), res9);
                            //res10 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i10 * n + j]), res10);
                            //res11 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i11 * n + j]), res11);
                            //res12 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i12 * n + j]), res12);
                            //res13 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i13 * n + j]), res13);
                            //res14 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[i14 * n + j]), res14);
                            #endregion
                            if(true)
                            #region ak
                            {
                                res0 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 0]), res0);
                                res1 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 1]), res1);
                                res2 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 2]), res2);
                                res3 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 3]), res3);
                                res4 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 4]), res4);
                                res5 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 5]), res5);
                                res6 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 6]), res6);
                                res7 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 7]), res7);
                                res8 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 8]), res8);
                                res9 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 9]), res9);
                                res10 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 10]), res10);
                                res11 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 11]), res11);
                                res12 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 12]), res12);
                                res13 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 13]), res13);
                                //res14 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 14]), res14);
                            }
                            #endregion

                            #region TransposeA
                            //a -> m x n
                            //res0 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i0]), res0);
                            //res1 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i1]), res1);
                            //res2 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i2]), res2);
                            //res3 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i3]), res3);
                            //res4 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i4]), res4);
                            //res5 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i5]), res5);
                            //res6 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i6]), res6);
                            //res7 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i7]), res7);
                            //res8 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i8]), res8);
                            //res9 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i9]), res9);
                            //res10 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i10]), res10);
                            //res11 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i11]), res11);
                            //res12 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i12]), res12);
                            //res13 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[j * m + i13]), res13);
                            #endregion
                        }

                        Avx2.Store(&ptr_c[i0 * p + k], res0);
                        Avx2.Store(&ptr_c[i1 * p + k], res1);
                        Avx2.Store(&ptr_c[i2 * p + k], res2);
                        Avx2.Store(&ptr_c[i3 * p + k], res3);
                        Avx2.Store(&ptr_c[i4 * p + k], res4);
                        Avx2.Store(&ptr_c[i5 * p + k], res5);
                        Avx2.Store(&ptr_c[i6 * p + k], res6);
                        Avx2.Store(&ptr_c[i7 * p + k], res7);
                        Avx2.Store(&ptr_c[i8 * p + k], res8);
                        Avx2.Store(&ptr_c[i9 * p + k], res9);
                        Avx2.Store(&ptr_c[i10 * p + k], res10);
                        Avx2.Store(&ptr_c[i11 * p + k], res11);
                        Avx2.Store(&ptr_c[i12 * p + k], res12);
                        Avx2.Store(&ptr_c[i13 * p + k], res13);
                        //Avx2.Store(&ptr_c[i14 * p + k], res14);
                    }//k < p remaining part

                }

            }
            ak.Dispose();
            bk.Dispose();
        }






        #region will be deleted
        /// <summary>
        /// result = Matmul(arr1, T(arr2)) Not ready! Do not use this function.
        /// </summary>
        /// <param name="arr1"></param>
        /// <param name="arr2"></param>
        /// <param name="result">Result matrix shouldn't be the same matrix with arr1 and arr2</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MatrixMultiplyWithTranspose(ref Matrix arr1, ref Matrix arr2, ref Matrix result)
        {
            long increment = 3;
            fixed (float* ptr_a = arr1.Array, ptr_b = arr2.Array, ptr_res = result.Array)
            {
                for (long ri = 0; ri < arr1.D1 / increment * increment; ri += increment)
                {
                    for (long rj = 0; rj < arr2.D1 / increment * increment; rj += increment)
                    {
                        Vector256<float> sum00 = new Vector256<float>();
                        Vector256<float> sum01 = new Vector256<float>();
                        Vector256<float> sum02 = new Vector256<float>();
                        Vector256<float> sum10 = new Vector256<float>();
                        Vector256<float> sum11 = new Vector256<float>();
                        Vector256<float> sum12 = new Vector256<float>();
                        Vector256<float> sum20 = new Vector256<float>();
                        Vector256<float> sum21 = new Vector256<float>();
                        Vector256<float> sum22 = new Vector256<float>();

                        long loci0 = ri * arr1.D2;
                        long loci1 = loci0 + arr1.D2;
                        long loci2 = loci1 + arr1.D2;

                        long locj0 = rj * arr2.D2;
                        long locj1 = locj0 + arr2.D2;
                        long locj2 = locj1 + arr2.D2;


                        Avx2.Prefetch0(&ptr_a[loci0]);
                        Avx2.Prefetch0(&ptr_a[loci1]);
                        Avx2.Prefetch0(&ptr_a[loci2]);
                        Avx2.Prefetch0(&ptr_b[locj0]);
                        Avx2.Prefetch0(&ptr_b[locj1]);
                        Avx2.Prefetch0(&ptr_b[locj2]);


                        for (long l = 0; l < arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l += Vector256<float>.Count)
                        {
                            Vector256<float> i = Avx2.LoadVector256(&ptr_a[loci0 + l]);
                            Vector256<float> j0 = Avx2.LoadVector256(&ptr_b[locj0 + l]);
                            sum00 = Fma.MultiplyAdd(j0, i, sum00);
                            Vector256<float> j1 = Avx2.LoadVector256(&ptr_b[locj1 + l]);
                            sum01 = Fma.MultiplyAdd(i, j1, sum01);
                            Vector256<float> j2 = Avx2.LoadVector256(&ptr_b[locj2 + l]);
                            sum02 = Fma.MultiplyAdd(i, j2, sum02);
                            i = Avx2.LoadVector256(&ptr_a[loci1 + l]);
                            sum10 = Fma.MultiplyAdd(i, j0, sum10);
                            sum11 = Fma.MultiplyAdd(i, j1, sum11);
                            sum12 = Fma.MultiplyAdd(i, j2, sum12);
                            i = Avx2.LoadVector256(&ptr_a[loci2 + l]);

                            sum20 = Fma.MultiplyAdd(i, j0, sum20);
                            sum21 = Fma.MultiplyAdd(i, j1, sum21);
                            sum22 = Fma.MultiplyAdd(i, j2, sum22);
                        }
                        
                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        float res00 = sum00.GetElement(0) + sum00.GetElement(4);
                        sum01 = Fma.HorizontalAdd(sum01, sum01);
                        sum01 = Fma.HorizontalAdd(sum01, sum01);
                        float res01 = sum01.GetElement(0) + sum01.GetElement(4);
                        sum02 = Fma.HorizontalAdd(sum02, sum02);
                        sum02 = Fma.HorizontalAdd(sum02, sum02);
                        float res02 = sum02.GetElement(0) + sum02.GetElement(4);

                        sum10 = Fma.HorizontalAdd(sum10, sum10);
                        sum10 = Fma.HorizontalAdd(sum10, sum10);
                        float res10 = sum10.GetElement(0) + sum10.GetElement(4);
                        sum11 = Fma.HorizontalAdd(sum11, sum11);
                        sum11 = Fma.HorizontalAdd(sum11, sum11);
                        float res11 = sum11.GetElement(0) + sum11.GetElement(4);
                        sum12 = Fma.HorizontalAdd(sum12, sum12);
                        sum12 = Fma.HorizontalAdd(sum12, sum12);
                        float res12 = sum12.GetElement(0) + sum12.GetElement(4);

                        sum20 = Fma.HorizontalAdd(sum20, sum20);
                        sum20 = Fma.HorizontalAdd(sum20, sum20);
                        float res20 = sum20.GetElement(0) + sum20.GetElement(4);
                        sum21 = Fma.HorizontalAdd(sum21, sum21);
                        sum21 = Fma.HorizontalAdd(sum21, sum21);
                        float res21 = sum21.GetElement(0) + sum21.GetElement(4);
                        sum22 = Fma.HorizontalAdd(sum22, sum22);
                        sum22 = Fma.HorizontalAdd(sum22, sum22);
                        float res22 = sum22.GetElement(0) + sum22.GetElement(4);

                        for (long l = arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l < arr2.D2; l++)
                        {
                            res00 += ptr_a[loci0 + l] * ptr_b[locj0 + l];
                            res01 += ptr_a[loci0 + l] * ptr_b[locj1 + l];
                            res02 += ptr_a[loci0 + l] * ptr_b[locj2 + l];
                            res10 += ptr_a[loci1 + l] * ptr_b[locj0 + l];
                            res11 += ptr_a[loci1 + l] * ptr_b[locj1 + l];
                            res12 += ptr_a[loci1 + l] * ptr_b[locj2 + l];
                            res20 += ptr_a[loci2 + l] * ptr_b[locj0 + l];
                            res21 += ptr_a[loci2 + l] * ptr_b[locj1 + l];
                            res22 += ptr_a[loci2 + l] * ptr_b[locj2 + l];
                        }

                        ptr_res[ri * result.D2 + rj] = res00;
                        ptr_res[ri * result.D2 + rj + 1] = res01;
                        ptr_res[ri * result.D2 + rj + 2] = res02;
                        ptr_res[(ri+1) * result.D2 + rj] = res10;
                        ptr_res[(ri + 1) * result.D2 + rj + 1] = res11;
                        ptr_res[(ri + 1) * result.D2 + rj + 2] = res12;
                        ptr_res[(ri + 2) * result.D2 + rj] = res20;
                        ptr_res[(ri + 2) * result.D2 + rj + 1] = res21;
                        ptr_res[(ri + 2) * result.D2 + rj + 2] = res22;
                    }
                    for (long rj = arr2.D1 / increment * increment; rj < arr2.D1; rj++)
                    {
                        Vector256<float> sum00 = new Vector256<float>();
                        Vector256<float> sum10 = new Vector256<float>();
                        Vector256<float> sum20 = new Vector256<float>();

                        long loci0 = ri * arr1.D2;
                        long loci1 = loci0 + arr1.D2;
                        long loci2 = loci1 + arr1.D2;


                        long locj0 = rj * arr2.D2;

                        for (long l = 0; l < arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l += Vector256<float>.Count)
                        {
                            Vector256<float> i0 = Avx2.LoadVector256(&ptr_a[loci0 + l]);
                            Vector256<float> i1 = Avx2.LoadVector256(&ptr_a[loci1 + l]);
                            Vector256<float> i2 = Avx2.LoadVector256(&ptr_a[loci2 + l]);
                            Vector256<float> j0 = Avx2.LoadVector256(&ptr_b[locj0 + l]);
                            sum00 = Fma.MultiplyAdd(i0, j0, sum00);
                            sum10 = Fma.MultiplyAdd(i1, j0, sum10);
                            sum20 = Fma.MultiplyAdd(i2, j0, sum20);
                        }

                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        float res00 = sum00.GetElement(0) + sum00.GetElement(4);

                        sum10 = Fma.HorizontalAdd(sum10, sum10);
                        sum10 = Fma.HorizontalAdd(sum10, sum10);
                        float res10 = sum10.GetElement(0) + sum10.GetElement(4);

                        sum20 = Fma.HorizontalAdd(sum20, sum20);
                        sum20 = Fma.HorizontalAdd(sum20, sum20);
                        float res20 = sum20.GetElement(0) + sum20.GetElement(4);

                        for (long l = arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l < arr2.D2; l++)
                        {
                            res00 += ptr_a[loci0 + l] * ptr_b[locj0 + l];
                            res10 += ptr_a[loci1 + l] * ptr_b[locj0 + l];
                            res20 += ptr_a[loci2 + l] * ptr_b[locj0 + l];
                        }
                        ptr_res[ri * result.D2 + rj] = res00;
                        ptr_res[(ri + 1) * result.D2 + rj] = res10;
                        ptr_res[(ri + 2) * result.D2 + rj] = res20;
                    }
                }

                for (long ri = arr1.D1 / increment * increment; ri < arr1.D1; ri++)
                {
                    for (long rj = 0; rj < arr2.D1 / increment * increment; rj += increment)
                    {
                        Vector256<float> sum00 = new Vector256<float>();
                        Vector256<float> sum01 = new Vector256<float>();
                        Vector256<float> sum02 = new Vector256<float>();

                        long loci0 = ri * arr1.D2;

                        long locj0 = rj * arr2.D2;
                        long locj1 = locj0 + arr2.D2;
                        long locj2 = locj1 + arr2.D2;

                        for (long l = 0; l < arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l += Vector256<float>.Count)
                        {
                            Vector256<float> i0 = Avx2.LoadVector256(&ptr_a[loci0 + l]);
                            Vector256<float> j0 = Avx2.LoadVector256(&ptr_b[locj0 + l]);
                            Vector256<float> j1 = Avx2.LoadVector256(&ptr_b[locj1 + l]);
                            Vector256<float> j2 = Avx2.LoadVector256(&ptr_b[locj2 + l]);
                            sum00 = Fma.MultiplyAdd(i0, j0, sum00);
                            sum01 = Fma.MultiplyAdd(i0, j1, sum01);
                            sum02 = Fma.MultiplyAdd(i0, j2, sum02);
                        }

                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        float res00 = sum00.GetElement(0) + sum00.GetElement(4);
                        sum01 = Fma.HorizontalAdd(sum01, sum01);
                        sum01 = Fma.HorizontalAdd(sum01, sum01);
                        float res01 = sum01.GetElement(0) + sum01.GetElement(4);
                        sum02 = Fma.HorizontalAdd(sum02, sum02);
                        sum02 = Fma.HorizontalAdd(sum02, sum02);
                        float res02 = sum02.GetElement(0) + sum02.GetElement(4);

                        for (long l = arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l < arr2.D2; l++)
                        {
                            res00 += ptr_a[loci0 + l] * ptr_b[locj0 + l];
                            res01 += ptr_a[loci0 + l] * ptr_b[locj1 + l];
                            res02 += ptr_a[loci0 + l] * ptr_b[locj2 + l];
                        }
                        ptr_res[ri * result.D2 + rj] = res00;
                        ptr_res[ri * result.D2 + rj + 1] = res01;
                        ptr_res[ri * result.D2 + rj + 2] = res02;
                    }
                    for (long rj = arr2.D1 / increment * increment; rj < arr2.D1; rj++)
                    {
                        Vector256<float> sum00 = new Vector256<float>();
                        long loci0 = ri * arr1.D2;
                        long locj0 = rj * arr2.D2;
                        for (long l = 0; l < arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l += Vector256<float>.Count)
                        {
                            Vector256<float> i0 = Avx2.LoadVector256(&ptr_a[loci0 + l]);
                            Vector256<float> j0 = Avx2.LoadVector256(&ptr_b[locj0 + l]);
                            sum00 = Fma.MultiplyAdd(i0, j0, sum00);
                        }
                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        sum00 = Fma.HorizontalAdd(sum00, sum00);
                        float res00 = sum00.GetElement(0) + sum00.GetElement(4);
                        for (long l = arr2.D2 / Vector256<float>.Count * Vector256<float>.Count; l < arr2.D2; l++)
                            res00 += ptr_a[loci0 + l] * ptr_b[locj0 + l];
                        ptr_res[ri * result.D2 + rj] = res00;
                    }
                }
            }
        }
        /// <summary>
        /// Not ready! Do not use this function.
        /// </summary>
        /// <param name="arr1"></param>
        /// <param name="arr2"></param>
        /// <param name="result"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MatrixMultiplyParallel(ref Matrix arr1, ref Matrix arr2, ref Matrix result)
        {
            int rowbyte = (arr1.D2 * 4);
            int L1 = 256 * 1024;
            long increment = ((L1 / rowbyte) / 2);//16+16 rows will be cached. 32*1000*4 byte = 32*4kb we have 256kb l1 
            long maxsize = arr1.D1;
            long maxsize2 = arr1.D2;

            int degree = 0;
            int num;
            ThreadPool.GetMinThreads(out degree, out num);
            long maxinccanbe = maxsize / degree / 4;
            increment = Math.Max(1, Math.Min(maxinccanbe, increment));
            fixed (float* ptr_a = arr1.Array, ptr_b = arr2.Array, ptr_res = result.Array)
            {
                
                PointerCarrier a = new PointerCarrier();
                a.ptr = ptr_a;
                PointerCarrier b = new PointerCarrier();
                b.ptr = ptr_b;
                PointerCarrier res = new PointerCarrier();
                res.ptr = ptr_res;

                Parallel.For(0, (increment -1+ maxsize)/ increment, new ParallelOptions() { MaxDegreeOfParallelism = degree }, (long ri) =>
                {
                   
                    for (long rj = 0; rj < maxsize; rj += increment)
                    {
                        for (long i = ri; i < ri + increment && i < maxsize; i++)
                            for (long j = rj; j < rj + increment && j < maxsize; j++)
                            {
                                res.ptr[i * maxsize2 + j] = (float)DotProductAVX(&a.ptr[i * maxsize2], &b.ptr[j * maxsize2], maxsize2);
                            }
                    }
                });

                //Parallel.For(0, maxsize, new ParallelOptions() { MaxDegreeOfParallelism = degree }, (int i) =>
                //{
                //    for (int j = 0; j < maxsize; j++)
                //    {
                //        res.ptr[i*maxsize2 + j] = (float)DotProductAVX(&a.ptr[i * maxsize2], &b.ptr[j * maxsize2], maxsize2);
                //    }
                //});

            }
        }
        #endregion
    }
}
