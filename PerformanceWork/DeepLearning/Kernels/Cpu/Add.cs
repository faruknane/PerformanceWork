using PerformanceWork.OptimizedNumerics;
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
        public static Tensor AddFloat(params Tensor[] tensors)
        {
            Tensor res = new Tensor(tensors[0].Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWiseAddAVX((float*)tensors[0].Array, (float*)tensors[1].Array, (float*)res.Array, res.Shape.TotalSize);

            for (int i = 2; i < tensors.Length; i++) //todo add Optimize here. 
                VectorizationFloat.ElementWiseAddAVX((float*)res.Array, (float*)tensors[i].Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

    }
}
