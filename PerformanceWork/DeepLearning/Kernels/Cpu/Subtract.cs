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
    public unsafe static partial class CpuKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SubtractFloat32(Tensor a, Tensor b)
        {
            Tensor res;
            if (a.Shape.TotalSize < b.Shape.TotalSize)
                res = new Tensor(b.Shape.Clone(), TensorConfig.Host_Float32);
            else
                res = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
            SubtractFloat32(res, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SubtractFloat32(Tensor res, Tensor a, Tensor b)
        {
            if (a.Shape.TotalSize > b.Shape.TotalSize)
            {
                long go = res.Shape.TotalSize / b.Shape.TotalSize * b.Shape.TotalSize;
                for (long i = 0; i < go; i += b.Shape.TotalSize)
                    VectorizationFloat.ElementWiseSubtractAVX((float*)a.Base.Array + i, (float*)b.Base.Array, (float*)res.Base.Array + i, b.Shape.TotalSize);

                if (go < res.Shape.TotalSize)
                    VectorizationFloat.ElementWiseSubtractAVX((float*)a.Base.Array + go, (float*)b.Base.Array, (float*)res.Base.Array + go, res.Shape.TotalSize - go);
            }
            else
            {
                long go = res.Shape.TotalSize / a.Shape.TotalSize * a.Shape.TotalSize;
                for (long i = 0; i < go; i += a.Shape.TotalSize)
                    VectorizationFloat.ElementWiseSubtractAVX((float*)a.Base.Array, (float*)b.Base.Array + i, (float*)res.Base.Array + i, a.Shape.TotalSize);

                if (go < res.Shape.TotalSize)
                    VectorizationFloat.ElementWiseSubtractAVX((float*)a.Base.Array, (float*)b.Base.Array + go, (float*)res.Base.Array + go, res.Shape.TotalSize - go);
            }

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SubtractFloat32_GetGradientA(Tensor s, Tensor a, bool generateseperately = false)
        {
            Tensor gradienta;
            if (a.Shape.TotalSize == s.Shape.TotalSize && !generateseperately)
                gradienta = s;
            else
            {
                gradienta = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);

                SubtractFloat32_GetGradientA(gradienta, s, a);
            }
            return gradienta;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SubtractFloat32_GetGradientA(Tensor gradienta, Tensor s, Tensor a)
        {
            long go = s.Shape.TotalSize / gradienta.Shape.TotalSize * gradienta.Shape.TotalSize;
            for (long i = 0; i < go; i += gradienta.Shape.TotalSize)
                if (i == 0)
                    VectorizationFloat.ElementWiseAssignAVX((float*)gradienta.Base.Array, (float*)s.Base.Array + i, gradienta.Shape.TotalSize);
                else
                    VectorizationFloat.ElementWiseAddAVX((float*)s.Base.Array + i, (float*)gradienta.Base.Array, (float*)gradienta.Base.Array, gradienta.Shape.TotalSize);

            if (go < s.Shape.TotalSize)
                VectorizationFloat.ElementWiseAddAVX((float*)s.Base.Array + go, (float*)gradienta.Base.Array, (float*)gradienta.Base.Array, s.Shape.TotalSize - go);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor SubtractFloat32_GetGradientB(Tensor s, Tensor a)
        {
            Tensor gradienta = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
            SubtractFloat32_GetGradientB(gradienta, s, a);
            return gradienta;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void SubtractFloat32_GetGradientB(Tensor gradienta, Tensor s, Tensor a)
        {
            long go = s.Shape.TotalSize / gradienta.Shape.TotalSize * gradienta.Shape.TotalSize;
            for (long i = 0; i < go; i += gradienta.Shape.TotalSize)
                if (i == 0)
                    VectorizationFloat.MakeNegativeAVX((float*)s.Base.Array + i, (float*)gradienta.Base.Array, gradienta.Shape.TotalSize);
                else
                    VectorizationFloat.ElementWiseSubtractAVX((float*)gradienta.Base.Array, (float*)s.Base.Array + i, (float*)gradienta.Base.Array, gradienta.Shape.TotalSize);

            if (go < s.Shape.TotalSize)
                VectorizationFloat.ElementWiseSubtractAVX((float*)gradienta.Base.Array, (float*)s.Base.Array + go, (float*)gradienta.Base.Array, s.Shape.TotalSize-go);
        }


    }
}
