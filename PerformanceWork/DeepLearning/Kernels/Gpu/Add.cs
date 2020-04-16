using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.DeepLearning.Kernels.Gpu
{
    public unsafe static partial class GpuKernels
    {
        public static Tensor AddFloat(params Tensor[] Terms)
        {
            Tensor res = new Tensor(Terms[0].Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
           
            return res;
        }

    }
}
