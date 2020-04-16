using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.DeepLearning.Kernels.Cpu
{
    public unsafe static partial class CpuKernels
    {
        public static Tensor AddFloat(params Tensor[] Terms)
        {
            Tensor res = new Tensor(Terms[0].Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWiseAddAVX((float*)Terms[0].Array, (float*)Terms[1].Array, (float*)res.Array, res.Shape.TotalSize);

            for (int i = 2; i < Terms.Length; i++) //todo add Optimize here. 
                VectorizationFloat.ElementWiseAddAVX((float*)res.Array, (float*)Terms[i].Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

    }
}
