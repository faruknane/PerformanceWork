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
        private static unsafe void ElementWiseDivideAVX(float val, float* ptr_a, float* ptr_res, int length)
        {
            float* ptr_val = &val;
            for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> v2 = Avx2.BroadcastScalarToVector256(ptr_val);
                Vector256<float> res = Avx2.Divide(v2, v1);
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
            {
                ptr_res[i] = *ptr_val / ptr_a[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseSetValueAVX(float[] x, float val, long length)
        {
            float* ptr_val = &val;
            fixed (float* ptr_a = x)
            {
                Vector256<float> v = Avx2.BroadcastScalarToVector256(ptr_val);
                for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                    Avx2.Store(&ptr_a[i], v);
                for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                    x[i] = val;
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseSetValueAVX(float* ptr_a, float val, long length)
        {
            float* ptr_val = &val;
            Vector256<float> v = Avx2.BroadcastScalarToVector256(ptr_val);
            for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                Avx2.Store(&ptr_a[i], v);
            for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                ptr_a[i] = val;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseAddAVX(float[] left, float[] right, float[] result, long length)
        {
            fixed (float* ptr_a = left, ptr_b = right, ptr_res = result)
            {
                for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    Vector256<float> res = Avx2.Add(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] + ptr_b[i];
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseAddAVX(float* ptr_a, float* ptr_b, float* ptr_res, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> res = Avx2.Add(v1, Avx2.LoadVector256(&ptr_b[i]));
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] + ptr_b[i];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseDivideAVX(float* ptr_a, float* ptr_b, float* ptr_res, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> res = Avx2.Divide(v1, Avx2.LoadVector256(&ptr_b[i]));
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i]/ptr_b[i];
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseAddAVX(float[] left, float right, float[] result, long length)
        {
            float* ptr_b = &right;
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            fixed (float* ptr_a = left, ptr_res = result)
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.BroadcastScalarToVector256(ptr_b);
                    Vector256<float> res = Avx2.Add(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] + *ptr_b;
                }
            }
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseSubtractAVX(float[] left, float[] right, float[] result, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            fixed (float* ptr_a = left, ptr_b = right, ptr_res = result)
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    Vector256<float> res = Avx2.Subtract(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] - ptr_b[i];
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseSubtractAVX(float* ptr_left, float* ptr_right, float* ptr_res, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_left[i]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_right[i]);
                Vector256<float> res = Avx2.Subtract(v1, v2);
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_left[i] - ptr_right[i];
            }
        }
        /// <summary>
        /// res = left - right*BetaB
        /// </summary>
        /// <param name="ptr_left"></param>
        /// <param name="ptr_right"></param>
        /// <param name="ptr_res"></param>
        /// <param name="length"></param>
        /// <param name="b"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseSubtractAVXBetaB(float* ptr_left, float* ptr_right, float* ptr_res, long length, float b)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_left[i]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_right[i]);
                v2 = Avx2.Multiply(v2, Avx2.BroadcastScalarToVector256(&b));
                Vector256<float> res = Avx2.Subtract(v1, v2);
                Avx2.Store(&ptr_res[i], res);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_left[i] - ptr_right[i]*b;
            }
        }
        /// <summary>
        /// res = left + right*BetaB
        /// </summary>
        /// <param name="ptr_left"></param>
        /// <param name="ptr_right"></param>
        /// <param name="ptr_res"></param>
        /// <param name="length"></param>
        /// <param name="b"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseAddAVXBetaB(float* ptr_left, float* ptr_right, float* ptr_res, long length, float b)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            Vector256<float> v3 = Avx2.BroadcastScalarToVector256(&b);
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_left[i]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_right[i]);
                v1 = Fma.MultiplyAdd(v2, v3, v1);
                Avx2.Store(&ptr_res[i], v1);
            }
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_left[i] + ptr_right[i]*b;
            }
        }

        /// <summary>
        /// result =  left Dot right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        /// <param name="length"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe float DotProductFMA(float[] left, float[] right, long length)
        {
            fixed (float* ptr_a = left, ptr_b = right)
            {
                long remain = length % Vector256<float>.Count;

                Vector256<float> sum = new Vector256<float>();
                long i = 0;
                while (i < length - remain)
                {
                    //Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    sum = Fma.MultiplyAdd(Avx2.LoadVector256(&ptr_a[i]), v2, sum);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe float DotProductFMA(float* ptr_a, float* ptr_b, long length)
        {
            long remain = length % Vector256<float>.Count;

            Vector256<float> sum = new Vector256<float>();
            long i = 0;
            while (i < length - remain)
            {
                //Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                sum = Fma.MultiplyAdd(Avx2.LoadVector256(&ptr_a[i]), v2, sum);
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
        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //public static unsafe float DotProductFMAParallel(float* ptr_a, float* ptr_b, long length)
        //{
        //    int degree = 0;
        //    int num;
        //    ThreadPool.GetMinThreads(out degree, out num);
        //    PointerCarrier carrierA = new PointerCarrier();
        //    PointerCarrier carrierB = new PointerCarrier();
        //    carrierA.ptr = ptr_a;
        //    carrierB.ptr = ptr_b;

        //    long[] blocks = new long[degree];

        //    for (int i = 0; i < degree; i++)
        //        blocks[i] = length / degree;

        //    blocks[degree-1] += length % degree;

        //    long[] starts = new long[degree];
        //    long sum = 0;
        //    for (int i = 0; i < degree; i++)
        //    {
        //        starts[i] = sum;
        //        sum += blocks[i];
        //    }

        //    float result = 0;
        //    object locker = new object();
        //    Parallel.For(0, degree, new ParallelOptions() { MaxDegreeOfParallelism = degree} ,(long threadno) =>
        //    {
        //        float ownres = DotProductFMA(carrierA.ptr + starts[threadno], carrierB.ptr + starts[threadno], blocks[threadno]);
        //        lock (locker)
        //        {
        //            result += ownres;
        //        }
        //    });
        //    return result;
        //}
        /// <summary>
        /// Tranpose of a 8x8 Matrix
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        /// <param name="length"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatrixTranspose8x8(float[] matrix, float[] result)
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseMultiplyAVX(float[] arr1, float[] arr2, float[] result, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            fixed (float* ptr_a = arr1, ptr_b = arr2, ptr_res = result)
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    Vector256<float> res = Avx2.Multiply(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] * ptr_b[i];
                }
            }
        }

        

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseMultiplyAVX(float[] arr1, float arr2, float[] result, long length)
        {
            float* ptr_b = &arr2;
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            fixed (float* ptr_a = arr1, ptr_res = result)
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.BroadcastScalarToVector256(ptr_b);
                    Vector256<float> res = Avx2.Multiply(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] * *ptr_b;
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseMultiplyAVX(float* ptr_a, float* ptr_b, float* ptr_res, long length)
        {
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            {
                for (long i = 0; i < l; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                    Vector256<float> res = Avx2.Multiply(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = l; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] * ptr_b[i];
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe float ElementWiseMultiplyAndReturnSum(float* ptr_a, float* ptr_b, float* ptr_res, long length)
        {
            Vector256<float> sum = new Vector256<float>();
            long l = length / Vector256<float>.Count * Vector256<float>.Count;
            for (long i = 0; i < l; i += Vector256<float>.Count)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                Vector256<float> res = Avx2.Multiply(v1, v2);
                sum = Avx2.Add(sum, res);
                Avx2.Store(&ptr_res[i], res);
            }

            float result = 0;
            sum = Fma.HorizontalAdd(sum, sum);
            sum = Fma.HorizontalAdd(sum, sum);
            result = sum.GetElement(0) + sum.GetElement(4);

            float remainingsum = 0;
            for (long i = l; i < length; i++)
            {
                ptr_res[i] = ptr_a[i] * ptr_b[i];
                remainingsum += ptr_res[i];
            }
            result += remainingsum;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseAddAVX(float* ptr_a, float right, float* ptr_res, long length)
        {
            float* ptr_b = &right;
            {
                for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.BroadcastScalarToVector256(ptr_b);
                    Vector256<float> res = Avx2.Add(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] + *ptr_b;
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ElementWiseMultiplyAVX(float* ptr_a, float arr2, float* ptr_res, long length)
        {
            float* ptr_b = &arr2;
            {
                for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                    Vector256<float> v2 = Avx2.BroadcastScalarToVector256(ptr_b);
                    Vector256<float> res = Avx2.Multiply(v1, v2);
                    Avx2.Store(&ptr_res[i], res);
                }
                for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                {
                    ptr_res[i] = ptr_a[i] * *ptr_b;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static void ElementWiseAssignAVX(float[] left, float[] right, long length)
        {
            fixed (float* ptr_left = left, ptr_right = right)
            {
                for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                {
                    Vector256<float> v = Avx2.LoadVector256(&ptr_right[i]);
                    Avx2.Store(&ptr_left[i], v);
                }
                for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                {
                    ptr_left[i] = ptr_right[i];
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static void ElementWiseAssignAVX(float* ptr_left, float* ptr_right, long length)
        {
            {
                for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                {
                    Vector256<float> v = Avx2.LoadVector256(&ptr_right[i]);
                    Avx2.Store(&ptr_left[i], v);
                }
                for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                {
                    ptr_left[i] = ptr_right[i];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static float SumOfVector(float* vector, long length)
        {
            long remain = length % Vector256<float>.Count;

            Vector256<float> sum = new Vector256<float>();
            long i = 0;
            while (i < length - remain)
            {
                Vector256<float> v1 = Avx2.LoadVector256(&vector[i]);
                sum = Avx2.Add(sum, v1);
                i += Vector256<float>.Count;
            }

            float result = 0;
            sum = Fma.HorizontalAdd(sum, sum);
            sum = Fma.HorizontalAdd(sum, sum);
            result = sum.GetElement(0) + sum.GetElement(4);
            
            float remainingsum = 0;
            for (i = length - remain; i < length; i++)
                remainingsum += vector[i];
            result += remainingsum;

            return result;
        }

        

        #endregion

        #region Unoptimized Methods
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe bool ElementWiseIsEqualsAVX(float[] arr1, float[] arr2, long length)
        {
            fixed (float* ptr_a = arr1, ptr_b = arr2)
            {
                for (long i = 0; i < length; i++)
                    if (ptr_a[i] != ptr_b[i])
                        return false;
                return true;
                //for (long i = 0; i < length / Vector256<float>.Count * Vector256<float>.Count; i += Vector256<float>.Count)
                //{
                //    Vector256<float> v1 = Avx2.LoadVector256(&ptr_a[i]);
                //    Vector256<float> v2 = Avx2.LoadVector256(&ptr_b[i]);
                //    Vector256<float> res = Avx2.Compare(v1, v2);
                //    Avx2.Store(&ptr_res[i], res);
                //}
                //for (long i = length / Vector256<float>.Count * Vector256<float>.Count; i < length; i++)
                //{
                //    ptr_res[i] = ptr_a[i] * ptr_b[i];
                //}
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe bool ElementWiseIsEqualsAVX(float* ptr_a, float* ptr_b, long length)
        {
                for (long i = 0; i < length; i++)
                    if (ptr_a[i] != ptr_b[i])
                        return false;
                return true;
           
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //public unsafe static void MatrixMultiplyMine(ref Matrix a, ref Matrix b, ref Matrix c)
        //{
        //    c.SetZero();
        //    long m = a.D1, n = a.D2, p = b.D2;
        //    Matrix bk = new Matrix(b.D1, b.D2);
        //    Matrix ak = new Matrix(a.D1, a.D2);
        //    long increment = 14;

        //    float * ptr_a = ak.Array, ptr_b = bk.Array, ptr_c = c.Array, ptr_real_a = a.Array, ptr_real_b = b.Array;

        //    {
        //        #region editing ak,bk array with pointers
        //        for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
        //        {
        //            long positionyforcreating = (k / Vector256<float>.Count) * Vector256<float>.Count * n;
        //            for (long j = 0; j < n; ++j)
        //            {
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 0];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 1];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 2];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 3];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 4];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 5];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 6];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 7];
        //                ++positionyforcreating;
        //            }
        //        }


        //        for (long fi = 0; fi < m / increment * increment; fi += increment)
        //        {
        //            for (long j = 0; j < n; ++j)
        //            {
        //                //ptr_a[i0 * n + j]
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 0] = ptr_real_a[(fi + 0) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 1] = ptr_real_a[(fi + 1) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 2] = ptr_real_a[(fi + 2) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 3] = ptr_real_a[(fi + 3) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 4] = ptr_real_a[(fi + 4) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 5] = ptr_real_a[(fi + 5) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 6] = ptr_real_a[(fi + 6) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 7] = ptr_real_a[(fi + 7) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 8] = ptr_real_a[(fi + 8) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 9] = ptr_real_a[(fi + 9) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 10] = ptr_real_a[(fi + 10) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 11] = ptr_real_a[(fi + 11) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 12] = ptr_real_a[(fi + 12) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 13] = ptr_real_a[(fi + 13) * n + j];
        //                //ptr_a[(fi / increment) * (increment * n) + j * increment + 14] = a[fi + 14, j];
        //            }
        //        }




        //        #endregion
        //        for (long fi = 0; fi < m / increment * increment; fi += increment)
        //        {
        //            long i0 = fi;
        //            long i1 = fi + 1;
        //            long i2 = fi + 2;
        //            long i3 = fi + 3;
        //            long i4 = fi + 4;
        //            long i5 = fi + 5;
        //            long i6 = fi + 6;
        //            long i7 = fi + 7;
        //            long i8 = fi + 8;
        //            long i9 = fi + 9;
        //            long i10 = fi + 10;
        //            long i11 = fi + 11;
        //            long i12 = fi + 12;
        //            long i13 = fi + 13;
        //            //long i14 = fi + 14;

        //            //we can cancel (/ Vector256<float>.Count * Vector256<float>.Count) below
        //            for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
        //            {
        //                // 4 x 8 result
        //                Vector256<float> res0 = new Vector256<float>(); //first row of the result
        //                Vector256<float> res1 = new Vector256<float>();
        //                Vector256<float> res2 = new Vector256<float>();
        //                Vector256<float> res3 = new Vector256<float>();
        //                Vector256<float> res4 = new Vector256<float>();
        //                Vector256<float> res5 = new Vector256<float>();
        //                Vector256<float> res6 = new Vector256<float>();
        //                Vector256<float> res7 = new Vector256<float>();
        //                Vector256<float> res8 = new Vector256<float>();
        //                Vector256<float> res9 = new Vector256<float>();
        //                Vector256<float> res10 = new Vector256<float>();
        //                Vector256<float> res11 = new Vector256<float>();
        //                Vector256<float> res12 = new Vector256<float>();
        //                Vector256<float> res13 = new Vector256<float>();
        //                //Vector256<float> res14 = new Vector256<float>();

        //                long positiony = (k / Vector256<float>.Count) * Vector256<float>.Count * n;
        //                long postionx = (fi / increment) * (increment * n);

        //                //Avx2.Prefetch0(&ptr_b[positiony]);
        //                //Avx2.Prefetch0(&j);
        //                for (long j = 0; j < n; ++j)
        //                {
        //                    Vector256<float> y = Avx2.LoadVector256(&ptr_b[positiony + j * Vector256<float>.Count]);
        //                    if (true)
        //                    {
        //                        res0 = Fma.MultiplyAdd(Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 0]),y, res0);
        //                        res1 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 1]), res1);
        //                        res2 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 2]), res2);
        //                        res3 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 3]), res3);
        //                        res4 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 4]), res4);
        //                        res5 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 5]), res5);
        //                        res6 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 6]), res6);
        //                        res7 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 7]), res7);
        //                        res8 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 8]), res8);
        //                        res9 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 9]), res9);
        //                        res10 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 10]), res10);
        //                        res11 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 11]), res11);
        //                        res12 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 12]), res12);
        //                        res13 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 13]), res13);
        //                        //res14 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 14]), res14);
        //                    }

        //                }

        //                Avx2.Store(&ptr_c[i0 * p + k], res0);
        //                Avx2.Store(&ptr_c[i1 * p + k], res1);
        //                Avx2.Store(&ptr_c[i2 * p + k], res2);
        //                Avx2.Store(&ptr_c[i3 * p + k], res3);
        //                Avx2.Store(&ptr_c[i4 * p + k], res4);
        //                Avx2.Store(&ptr_c[i5 * p + k], res5);
        //                Avx2.Store(&ptr_c[i6 * p + k], res6);
        //                Avx2.Store(&ptr_c[i7 * p + k], res7);
        //                Avx2.Store(&ptr_c[i8 * p + k], res8);
        //                Avx2.Store(&ptr_c[i9 * p + k], res9);
        //                Avx2.Store(&ptr_c[i10 * p + k], res10);
        //                Avx2.Store(&ptr_c[i11 * p + k], res11);
        //                Avx2.Store(&ptr_c[i12 * p + k], res12);
        //                Avx2.Store(&ptr_c[i13 * p + k], res13);
        //                //Avx2.Store(&ptr_c[i14 * p + k], res14);
        //            }//k < p remaining part
        //            for (long k = p / Vector256<float>.Count * Vector256<float>.Count; k < p; k++)
        //            {
        //                long postionx = (fi / increment) * (increment * n);
        //                for (int j = 0; j < n; j++)
        //                {
        //                    ptr_c[i0 * p + k] += ptr_a[postionx + j * increment + 0] * ptr_real_b[j * p + k];
        //                    ptr_c[i1 * p + k] += ptr_a[postionx + j * increment + 1] * ptr_real_b[j * p + k];
        //                    ptr_c[i2 * p + k] += ptr_a[postionx + j * increment + 2] * ptr_real_b[j * p + k];
        //                    ptr_c[i3 * p + k] += ptr_a[postionx + j * increment + 3] * ptr_real_b[j * p + k];
        //                    ptr_c[i4 * p + k] += ptr_a[postionx + j * increment + 4] * ptr_real_b[j * p + k];
        //                    ptr_c[i5 * p + k] += ptr_a[postionx + j * increment + 5] * ptr_real_b[j * p + k];
        //                    ptr_c[i6 * p + k] += ptr_a[postionx + j * increment + 6] * ptr_real_b[j * p + k];
        //                    ptr_c[i7 * p + k] += ptr_a[postionx + j * increment + 7] * ptr_real_b[j * p + k];
        //                    ptr_c[i8 * p + k] += ptr_a[postionx + j * increment + 8] * ptr_real_b[j * p + k];
        //                    ptr_c[i9 * p + k] += ptr_a[postionx + j * increment + 9] * ptr_real_b[j * p + k];
        //                    ptr_c[i10 * p + k] += ptr_a[postionx + j * increment + 10] * ptr_real_b[j * p + k];
        //                    ptr_c[i11 * p + k] += ptr_a[postionx + j * increment + 11] * ptr_real_b[j * p + k];
        //                    ptr_c[i12 * p + k] += ptr_a[postionx + j * increment + 12] * ptr_real_b[j * p + k];
        //                    ptr_c[i13 * p + k] += ptr_a[postionx + j * increment + 13] * ptr_real_b[j * p + k];
        //                }

        //            }
        //        }
        //        for (long fi = m / increment * increment; fi < m; fi++)
        //        {
        //            long i0 = fi;
        //            for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
        //            {
        //                // 4 x 8 result
        //                Vector256<float> res0 = new Vector256<float>(); //first row of the result

        //                long positiony = (k / Vector256<float>.Count) * Vector256<float>.Count * n;

        //                //Avx2.Prefetch0(&ptr_b[positiony]);
        //                //Avx2.Prefetch0(&j);
        //                for (long j = 0; j < n; ++j)
        //                {
        //                    Vector256<float> y = Avx2.LoadVector256(&ptr_b[positiony + j * Vector256<float>.Count]);
        //                    if (true)
        //                    {
        //                        res0 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_real_a[fi * n + j]), res0);
        //                    }
        //                }
        //                Avx2.Store(&ptr_c[i0 * p + k], res0);
        //            }//k < p remaining part
        //            for (int j = 0; j < n; j++)
        //            {
        //                for (long k = p / Vector256<float>.Count * Vector256<float>.Count; k < p; k++)
        //                {
        //                    ptr_c[fi * p + k] += ptr_real_a[fi * n + j] * ptr_real_b[j * p + k];
        //                }
        //            }

        //        }
        //    }
        //    ak.Dispose();
        //    bk.Dispose();
        //}
        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //public unsafe static void MatrixMultiplyMine(float* a, long aD1, long aD2, float* b, long bD1, long bD2, float* c)
        //{
        //    Vectorization.ElementWiseSetValueAVX(c, 0, aD1 * bD2);
        //    long m = aD1, n = aD2, p = bD2;
        //    Matrix bk = new Matrix((int)bD1, (int)bD2);
        //    Matrix ak = new Matrix((int)aD1, (int)aD2);
        //    long increment = 14;

        //    float* ptr_real_a = a;
        //    float* ptr_real_b = b;
        //    float* ptr_c = c;
        //    float* ptr_a = ak.Array;
        //    float* ptr_b = bk.Array;
        //    {
        //        #region editing ak,bk array with pointers
        //        for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
        //        {
        //            long positionyforcreating = (k / Vector256<float>.Count) * Vector256<float>.Count * n;
        //            for (long j = 0; j < n; ++j)
        //            {
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 0];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 1];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 2];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 3];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 4];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 5];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 6];
        //                ++positionyforcreating;
        //                ptr_b[positionyforcreating] = ptr_real_b[j * p + k + 7];
        //                ++positionyforcreating;
        //            }
        //        }


        //        for (long fi = 0; fi < m / increment * increment; fi += increment)
        //        {
        //            for (long j = 0; j < n; ++j)
        //            {
        //                //ptr_a[i0 * n + j]
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 0] = ptr_real_a[(fi + 0) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 1] = ptr_real_a[(fi + 1) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 2] = ptr_real_a[(fi + 2) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 3] = ptr_real_a[(fi + 3) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 4] = ptr_real_a[(fi + 4) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 5] = ptr_real_a[(fi + 5) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 6] = ptr_real_a[(fi + 6) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 7] = ptr_real_a[(fi + 7) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 8] = ptr_real_a[(fi + 8) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 9] = ptr_real_a[(fi + 9) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 10] = ptr_real_a[(fi + 10) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 11] = ptr_real_a[(fi + 11) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 12] = ptr_real_a[(fi + 12) * n + j];
        //                ptr_a[(fi / increment) * (increment * n) + j * increment + 13] = ptr_real_a[(fi + 13) * n + j];
        //                //ptr_a[(fi / increment) * (increment * n) + j * increment + 14] = a[fi + 14, j];
        //            }
        //        }




        //        #endregion
        //        for (long fi = 0; fi < m / increment * increment; fi += increment)
        //        {
        //            long i0 = fi;
        //            long i1 = fi + 1;
        //            long i2 = fi + 2;
        //            long i3 = fi + 3;
        //            long i4 = fi + 4;
        //            long i5 = fi + 5;
        //            long i6 = fi + 6;
        //            long i7 = fi + 7;
        //            long i8 = fi + 8;
        //            long i9 = fi + 9;
        //            long i10 = fi + 10;
        //            long i11 = fi + 11;
        //            long i12 = fi + 12;
        //            long i13 = fi + 13;
        //            //long i14 = fi + 14;

        //            //we can cancel (/ Vector256<float>.Count * Vector256<float>.Count) below
        //            for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
        //            {
        //                // 4 x 8 result
        //                Vector256<float> res0 = new Vector256<float>(); //first row of the result
        //                Vector256<float> res1 = new Vector256<float>();
        //                Vector256<float> res2 = new Vector256<float>();
        //                Vector256<float> res3 = new Vector256<float>();
        //                Vector256<float> res4 = new Vector256<float>();
        //                Vector256<float> res5 = new Vector256<float>();
        //                Vector256<float> res6 = new Vector256<float>();
        //                Vector256<float> res7 = new Vector256<float>();
        //                Vector256<float> res8 = new Vector256<float>();
        //                Vector256<float> res9 = new Vector256<float>();
        //                Vector256<float> res10 = new Vector256<float>();
        //                Vector256<float> res11 = new Vector256<float>();
        //                Vector256<float> res12 = new Vector256<float>();
        //                Vector256<float> res13 = new Vector256<float>();
        //                //Vector256<float> res14 = new Vector256<float>();

        //                long positiony = (k / Vector256<float>.Count) * Vector256<float>.Count * n;
        //                long postionx = (fi / increment) * (increment * n);

        //                //Avx2.Prefetch0(&ptr_b[positiony]);
        //                //Avx2.Prefetch0(&j);
        //                for (long j = 0; j < n; ++j)
        //                {
        //                    Vector256<float> y = Avx2.LoadVector256(&ptr_b[positiony + j * Vector256<float>.Count]);
        //                    if (true)
        //                    {
        //                        res0 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 0]), res0);
        //                        res1 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 1]), res1);
        //                        res2 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 2]), res2);
        //                        res3 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 3]), res3);
        //                        res4 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 4]), res4);
        //                        res5 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 5]), res5);
        //                        res6 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 6]), res6);
        //                        res7 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 7]), res7);
        //                        res8 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 8]), res8);
        //                        res9 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 9]), res9);
        //                        res10 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 10]), res10);
        //                        res11 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 11]), res11);
        //                        res12 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 12]), res12);
        //                        res13 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 13]), res13);
        //                        //res14 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_a[postionx + j * increment + 14]), res14);
        //                    }

        //                }

        //                Avx2.Store(&ptr_c[i0 * p + k], res0);
        //                Avx2.Store(&ptr_c[i1 * p + k], res1);
        //                Avx2.Store(&ptr_c[i2 * p + k], res2);
        //                Avx2.Store(&ptr_c[i3 * p + k], res3);
        //                Avx2.Store(&ptr_c[i4 * p + k], res4);
        //                Avx2.Store(&ptr_c[i5 * p + k], res5);
        //                Avx2.Store(&ptr_c[i6 * p + k], res6);
        //                Avx2.Store(&ptr_c[i7 * p + k], res7);
        //                Avx2.Store(&ptr_c[i8 * p + k], res8);
        //                Avx2.Store(&ptr_c[i9 * p + k], res9);
        //                Avx2.Store(&ptr_c[i10 * p + k], res10);
        //                Avx2.Store(&ptr_c[i11 * p + k], res11);
        //                Avx2.Store(&ptr_c[i12 * p + k], res12);
        //                Avx2.Store(&ptr_c[i13 * p + k], res13);
        //                //Avx2.Store(&ptr_c[i14 * p + k], res14);
        //            }//k < p remaining part
        //            for (long k = p / Vector256<float>.Count * Vector256<float>.Count; k < p; k++)
        //            {
        //                long postionx = (fi / increment) * (increment * n);
        //                for (int j = 0; j < n; j++)
        //                {
        //                    ptr_c[i0 * p + k] += ptr_a[postionx + j * increment + 0] * ptr_real_b[j * p + k];
        //                    ptr_c[i1 * p + k] += ptr_a[postionx + j * increment + 1] * ptr_real_b[j * p + k];
        //                    ptr_c[i2 * p + k] += ptr_a[postionx + j * increment + 2] * ptr_real_b[j * p + k];
        //                    ptr_c[i3 * p + k] += ptr_a[postionx + j * increment + 3] * ptr_real_b[j * p + k];
        //                    ptr_c[i4 * p + k] += ptr_a[postionx + j * increment + 4] * ptr_real_b[j * p + k];
        //                    ptr_c[i5 * p + k] += ptr_a[postionx + j * increment + 5] * ptr_real_b[j * p + k];
        //                    ptr_c[i6 * p + k] += ptr_a[postionx + j * increment + 6] * ptr_real_b[j * p + k];
        //                    ptr_c[i7 * p + k] += ptr_a[postionx + j * increment + 7] * ptr_real_b[j * p + k];
        //                    ptr_c[i8 * p + k] += ptr_a[postionx + j * increment + 8] * ptr_real_b[j * p + k];
        //                    ptr_c[i9 * p + k] += ptr_a[postionx + j * increment + 9] * ptr_real_b[j * p + k];
        //                    ptr_c[i10 * p + k] += ptr_a[postionx + j * increment + 10] * ptr_real_b[j * p + k];
        //                    ptr_c[i11 * p + k] += ptr_a[postionx + j * increment + 11] * ptr_real_b[j * p + k];
        //                    ptr_c[i12 * p + k] += ptr_a[postionx + j * increment + 12] * ptr_real_b[j * p + k];
        //                    ptr_c[i13 * p + k] += ptr_a[postionx + j * increment + 13] * ptr_real_b[j * p + k];
        //                }

        //            }
        //        }
        //        for (long fi = m / increment * increment; fi < m; fi++)
        //        {
        //            long i0 = fi;
        //            for (long k = 0; k < p / Vector256<float>.Count * Vector256<float>.Count; k += Vector256<float>.Count)
        //            {
        //                // 4 x 8 result
        //                Vector256<float> res0 = new Vector256<float>(); //first row of the result

        //                long positiony = (k / Vector256<float>.Count) * Vector256<float>.Count * n;

        //                //Avx2.Prefetch0(&ptr_b[positiony]);
        //                //Avx2.Prefetch0(&j);
        //                for (long j = 0; j < n; ++j)
        //                {
        //                    Vector256<float> y = Avx2.LoadVector256(&ptr_b[positiony + j * Vector256<float>.Count]);
        //                    if (true)
        //                    {
        //                        res0 = Fma.MultiplyAdd(y, Avx2.BroadcastScalarToVector256(&ptr_real_a[fi * n + j]), res0);
        //                    }
        //                }
        //                Avx2.Store(&ptr_c[i0 * p + k], res0);
        //            }//k < p remaining part
        //            for (int j = 0; j < n; j++)
        //            {
        //                for (long k = p / Vector256<float>.Count * Vector256<float>.Count; k < p; k++)
        //                {
        //                    ptr_c[fi * p + k] += ptr_real_a[fi * n + j] * ptr_real_b[j * p + k];
        //                }
        //            }

        //        }
        //    }
        //    ak.Dispose();
        //    bk.Dispose();
        //}

        #endregion

        #region Unfinished Methods

        #endregion

    }
}
