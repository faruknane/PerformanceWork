using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor ShrinkFloat32(Tensor v, Shape thisShape, Shape term0, Shape Divisor)
        {
            Tensor res = new Tensor(thisShape.Clone(), TensorConfig.Host_Float32);
            ShrinkFloat32(res, v, thisShape, term0, Divisor);
            return res;
        } 
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void ShrinkFloat32(Tensor res, Tensor v, Shape thisShape, Shape term0, Shape Divisor)
        {
            res.SetFloat(0);

            float* ptrres = (float*)res.Array;
            float* ptrv = (float*)v.Array;

            Index iterator = new Index(term0);

            for (int i = 0; i < iterator.N; i++)
                iterator.Indices[i] = 0;

            for (int h = 0; h < term0.TotalSize; h++)
            {
                long indexs = 0;

                for (int i = iterator.N - 1; i >= 0; i--)
                {
                    if (iterator.Indices[i] == term0[i])
                    {
                        iterator.Indices[i] = 0;
                        iterator.Indices[i - 1]++;
                    }
                    indexs += (iterator.Indices[i] / Divisor[i]) * thisShape.Multiplied[i + 1];
                }
                ptrres[indexs] += ptrv[h];
                iterator.Indices[iterator.N - 1]++;
            }
        }
    }
}
