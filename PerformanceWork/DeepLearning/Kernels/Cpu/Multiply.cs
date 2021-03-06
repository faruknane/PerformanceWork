﻿using PerformanceWork.OptimizedNumerics;
using PerformanceWork.OptimizedNumerics.Tensors;
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
        public static Tensor MultiplyFloat32(Tensor a, Tensor b)
        {
            Tensor res;
            if (a.Shape.TotalSize < b.Shape.TotalSize)
                res = new Tensor(b.Shape.Clone(), TensorConfig.Host_Float32);
            else
                res = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
            MultiplyFloat32(res, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat32(Tensor res, Tensor a, Tensor b)
        {
            if (a.Shape.TotalSize > b.Shape.TotalSize)
            {
                Tensor temp = a;
                a = b;
                b = temp;
            }

            long go = res.Shape.TotalSize / a.Shape.TotalSize * a.Shape.TotalSize;
            for (long i = 0; i < go; i += a.Shape.TotalSize)
                VectorizationFloat.ElementWiseMultiplyAVX((float*)a.Base.Array, (float*)b.Base.Array + i, (float*)res.Base.Array + i, a.Shape.TotalSize);

            if (go < res.Shape.TotalSize)
                VectorizationFloat.ElementWiseMultiplyAVX((float*)a.Base.Array, (float*)b.Base.Array + go, (float*)res.Base.Array + go, res.Shape.TotalSize - go);

        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MultiplyFloat32_GetGradientA(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
            MultiplyFloat32_GetGradientA(res, s, a, b);
            return res;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void MultiplyFloat32_GetGradientA(Tensor gradienta, Tensor s, Tensor a, Tensor b)
        {
            if (s.Shape.TotalSize == a.Shape.TotalSize)
            {
                long go = s.Shape.TotalSize / b.Shape.TotalSize * b.Shape.TotalSize;
                for (long i = 0; i < go; i += b.Shape.TotalSize)
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Base.Array + i, (float*)b.Base.Array, (float*)gradienta.Base.Array + i, b.Shape.TotalSize);
                if (go < s.Shape.TotalSize)
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Base.Array + go, (float*)b.Base.Array, (float*)gradienta.Base.Array + go, s.Shape.TotalSize - go);
            }
            else if (s.Shape.TotalSize == b.Shape.TotalSize)
            {
                long go = s.Shape.TotalSize / a.Shape.TotalSize * a.Shape.TotalSize;
                for (long i = 0; i < go; i += a.Shape.TotalSize)
                    if (i == 0)
                        VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Base.Array, (float*)b.Base.Array, (float*)gradienta.Base.Array, gradienta.Shape.TotalSize);
                    else
                        VectorizationFloat.ElementWiseFMA((float*)s.Base.Array + i, (float*)b.Base.Array + i, (float*)gradienta.Base.Array, (float*)gradienta.Base.Array, gradienta.Shape.TotalSize);

                if (go < s.Shape.TotalSize)
                    VectorizationFloat.ElementWiseFMA((float*)s.Base.Array + go, (float*)b.Base.Array + go, (float*)gradienta.Base.Array, (float*)gradienta.Base.Array, s.Shape.TotalSize - go);
            }
            else
                throw new Exception("Impossible reagion MultiplyFloat32_GetGradientA!");
        }


    }
}