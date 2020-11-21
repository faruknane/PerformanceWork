﻿using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Index = PerformanceWork.OptimizedNumerics.Index;

namespace PerformanceWork.DeepLearning.Kernels.Cpu
{
    public unsafe partial class CpuKernels
    {
        /// <summary>
        /// Calculates the gradient of the Tensor to be expanded.
        /// </summary>
        /// <param name="s">Gradient tensor</param>
        /// <param name="multiplier">Shape indicates how much to expand, each element of the shape should be more than or equal to 1</param>
        /// <returns>The gradient of the Tensor to be expanded</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor ExpandFloat_GetGradient_0(Tensor s, Shape thisShape, Shape term0, Shape Multiplier)
        {
            Tensor combined = new Tensor(term0.Clone(), DeviceConfig.Host_Float);
            combined.SetFloat(0);

            float* ptrcombined = (float*)combined.Array;
            float* ptrs = (float*)s.Array;

            if (Multiplier.N == 2 && Multiplier[1] == 1)
            {
                for (int i = 0; i < Multiplier[0]; i++)
                {
                    float* me = ((float*)s.Array) + i * term0.TotalSize;
                    VectorizationFloat.ElementWiseAddAVX((float*)combined.Array, me, (float*)combined.Array, term0.TotalSize);
                }
            }
            else
            {
                Index iterator = new Index(thisShape);

                iterator.SetZero();

                for (int h = 0; h < thisShape.TotalSize; h++)
                {

                    int indexs = 0;

                    for (int i = iterator.N - 1; i >= 0; i--)
                    {
                        if (iterator.Indices[i] == thisShape[i])
                        {
                            iterator.Indices[i] = 0;
                            iterator.Indices[i - 1]++;
                        }
                        indexs += (iterator.Indices[i] / Multiplier[i]) * term0.Multiplied[i + 1];
                    }

                    ptrcombined[indexs] += ptrs[h];
                    iterator.Indices[iterator.N - 1]++;
                }
            }

            return combined;
        }

        /// <summary>
        /// Expands the Tensor given.
        /// </summary>
        /// <param name="res">Tensor to be expanded</param>
        /// <param name="multiplier">Shape indicates how much to expand, each element of the shape should be more than or equal to 1</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor ExpandFloat(Tensor v, Shape thisShape, Shape term0, Shape Multiplier)
        {
            Tensor res = new Tensor(thisShape.Clone(), DeviceConfig.Host_Float);

            float* ptrres = (float*)res.Array;
            float* ptrv = (float*)v.Array;

            if (Multiplier.N == 2 && Multiplier[1] == 1)
            {
                for (int i = 0; i < Multiplier[0]; i++)
                {
                    float* me = ((float*)res.Array) + i * term0.TotalSize;
                    VectorizationFloat.ElementWiseAssignAVX(me, (float*)v.Array, term0.TotalSize);
                }
            }
            else
            {

                Index iterator = new Index(res.Shape);

                for (int i = 0; i < iterator.N; i++)
                    iterator.Indices[i] = 0;

                for (int h = 0; h < res.Shape.TotalSize; h++)
                {
                    int indexs = 0;

                    for (int i = iterator.N - 1; i >= 0; i--)
                    {
                        if (iterator.Indices[i] == res.Shape[i])
                        {
                            iterator.Indices[i] = 0;
                            iterator.Indices[i - 1]++;
                        }
                        indexs += (iterator.Indices[i] / Multiplier[i]) * v.Shape.Multiplied[i + 1];
                    }
                    ptrres[h] = ptrv[indexs];
                    iterator.Indices[iterator.N - 1]++;
                }
            }
            return res;
        }

    }
}

//public static Tensor ExpandFloat(Tensor v, Shape thisShape, Shape term0, Shape Multiplier)
//{
//    Tensor res = new Tensor(thisShape.Clone(), DataType.Type.Float, DeviceIndicator.Host());

//    float* ptrres = (float*)res.Array;
//    float* ptrv = (float*)v.Array;

//    Index iterator = Index.NewIndex(res.Shape);

//    for (int i = 0; i < iterator.N; i++)
//        iterator.Indices[i] = 0;

//    for (int h = 0; h < res.Shape.TotalSize; h++)
//    {
//        int indexs = 0;

//        for (int i = iterator.N - 1; i >= 0; i--)
//        {
//            if (iterator.Indices[i] == res.Shape[i])
//            {
//                iterator.Indices[i] = 0;
//                iterator.Indices[i - 1]++;
//            }
//            indexs += (iterator.Indices[i] / Multiplier[i]) * v.Shape.Multiplied[i + 1];
//        }
//        ptrres[h] = ptrv[indexs];
//        iterator.Indices[iterator.N - 1]++;
//    }
//    Index.Return(iterator);

//    return res;
//}

