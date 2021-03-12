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
    /// Contains all of the Cpu kernels to process tensors.
    /// </summary>
    public unsafe static partial class CpuKernels
    {
        //todo add tensor a and b gradients.
        /// <summary>
        /// Returns the sum of tensors.
        /// </summary>
        /// <param name="tensors">Tensors to be summed</param>
        /// <returns>The sum of tensors</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor AddFloat32(Tensor a, Tensor b)
        {
            Tensor res;
            if (a.Shape.TotalSize < b.Shape.TotalSize)
                res = new Tensor(b.Shape.Clone(), TensorConfig.Host_Float32);
            else
                res = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
            AddFloat32(res, a, b);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32(Tensor res, Tensor a, Tensor b)
        {
            if (a.Shape.TotalSize > b.Shape.TotalSize)
            {
                Tensor temp = a;
                a = b;
                b = temp;
            }

            long go = res.Shape.TotalSize / a.Shape.TotalSize * a.Shape.TotalSize;
            for (long i = 0; i < go; i += a.Shape.TotalSize)
                VectorizationFloat.ElementWiseAddAVX((float*)a.Base.Array, (float*)b.Base.Array + i, (float*)res.Base.Array + i, a.Shape.TotalSize);

            if (go < res.Shape.TotalSize)
                VectorizationFloat.ElementWiseAddAVX((float*)a.Base.Array, (float*)b.Base.Array + go, (float*)res.Base.Array + go, res.Shape.TotalSize - go);

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor AddFloat32_GetGradient(Tensor s, Tensor a, bool generateseperately = false)
        {
            Tensor gradienta;
            if (a.Shape.TotalSize == s.Shape.TotalSize && !generateseperately)
                gradienta = s;
            else
            {
                gradienta = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);

                AddFloat32_GetGradient(gradienta, s, a);
            }
            return gradienta;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32_GetGradient(Tensor gradienta, Tensor s, Tensor a)
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



        //buranın altı gereksiz aslında
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static (Tensor, Tensor) AddFloat32_GetGradients(Tensor s, Tensor a, Tensor b, bool generateseperately = false)
        {
            return (AddFloat32_GetGradient(s, a, generateseperately), AddFloat32_GetGradient(s, b, generateseperately));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32_GetGradients(Tensor gradients_a, Tensor gradients_b, Tensor s, Tensor a, Tensor b)
        {
            AddFloat32_GetGradient(gradients_a, s, a);
            AddFloat32_GetGradient(gradients_b, s, b);
        }

        //no need for tensor array
        /// <summary>
        /// Returns the sum of tensors.
        /// </summary>
        /// <param name="tensors">Tensors to be summed</param>
        /// <returns>The sum of tensors</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor AddFloat32(Tensor[] tensors)
        {
            long size = -1;
            int index = -1;


            for (int i = 0; i < tensors.Length; i++)
                if (size < tensors[i].Shape.TotalSize)
                {
                    size = tensors[i].Shape.TotalSize;
                    index = i;
                }

            Tensor res = new Tensor(tensors[index].Shape.Clone(), TensorConfig.Host_Float32);
            AddFloat32(res, tensors, index);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32(Tensor res, Tensor[] tensors, int index = -1)
        {
            if (index == -1)
            {
                long size = -1;

                for (int i = 0; i < tensors.Length; i++)
                    if (size < tensors[i].Shape.TotalSize)
                    {
                        size = tensors[i].Shape.TotalSize;
                        index = i;
                    }
            }

            int index2 = 0;
            if (index == index2)
                index2++;

            AddFloat32(res, tensors[index], tensors[index2]);

            for (int i = 0; i < tensors.Length; i++)
                if (i != index && i != index2)
                {
                    AddFloat32(res, res, tensors[i]);
                }

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor[] AddFloat32_GetGradients(Tensor s, Tensor[] tensors, bool generateseperately = false)
        {
            Tensor[] gradients = new Tensor[tensors.Length];

            Dictionary<long, Tensor> dict = new Dictionary<long, Tensor>();
            dict[s.Shape.TotalSize] = s;

            for (int j = 0; j < gradients.Length; j++)
            {
                Tensor a = tensors[j];
                if (!generateseperately && dict.ContainsKey(a.Shape.TotalSize))
                    gradients[j] = dict[a.Shape.TotalSize];
                else
                {
                    Tensor gradient = new Tensor(a.Shape.Clone(), TensorConfig.Host_Float32);
                    //gradient.SetValue(0);

                    long go = s.Shape.TotalSize / gradient.Shape.TotalSize * gradient.Shape.TotalSize;
                    for (long i = 0; i < go; i += gradient.Shape.TotalSize)
                        if (go == 0)
                            VectorizationFloat.ElementWiseAssignAVX((float*)gradient.Base.Array, (float*)s.Base.Array + i, gradient.Shape.TotalSize);
                        else
                            VectorizationFloat.ElementWiseAddAVX((float*)s.Base.Array + i, (float*)gradient.Base.Array, (float*)gradient.Base.Array, gradient.Shape.TotalSize);

                    if (go < s.Shape.TotalSize)
                        VectorizationFloat.ElementWiseAddAVX((float*)s.Base.Array + go, (float*)gradient.Base.Array, (float*)gradient.Base.Array, s.Shape.TotalSize - go);

                }
            }

            return gradients;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32_GetGradients(Tensor[] gradients, Tensor s, Tensor[] tensors)
        {
            for (int j = 0; j < gradients.Length; j++)
            {
                Tensor gradient = gradients[j];
                //gradient.SetValue(0);

                long go = s.Shape.TotalSize / gradient.Shape.TotalSize * gradient.Shape.TotalSize;
                for (long i = 0; i < go; i += gradient.Shape.TotalSize)
                    if (go == 0)
                        VectorizationFloat.ElementWiseAssignAVX((float*)gradient.Base.Array, (float*)s.Base.Array + i, gradient.Shape.TotalSize);
                    else
                        VectorizationFloat.ElementWiseAddAVX((float*)s.Base.Array + i, (float*)gradient.Base.Array, (float*)gradient.Base.Array, gradient.Shape.TotalSize);

                if (go < s.Shape.TotalSize)
                    VectorizationFloat.ElementWiseAddAVX((float*)s.Base.Array + go, (float*)gradient.Base.Array, (float*)gradient.Base.Array, s.Shape.TotalSize - go);
            }
        }

    }
}
