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
        /// <summary>
        /// Returns the sum of tensors.
        /// </summary>
        /// <param name="tensors">Tensors to be summed</param>
        /// <returns>The sum of tensors</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor AddFloat32(params Tensor[] tensors)
        {
            Tensor res = new Tensor(tensors[0].Shape.Clone(), TensorConfig.Host_Float32);
            AddFloat32(res, tensors); 
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void AddFloat32(Tensor res, params Tensor[] tensors)
        {            
            VectorizationFloat.ElementWiseAddAVX((float*)tensors[0].Base.Array, (float*)tensors[1].Base.Array, (float*)res.Base.Array, res.Shape.TotalSize);

            for (int i = 2; i < tensors.Length; i++) //todo add Optimize here. 
                VectorizationFloat.ElementWiseAddAVX((float*)res.Base.Array, (float*)tensors[i].Base.Array, (float*)res.Base.Array, res.Shape.TotalSize);
        }

    }
}
