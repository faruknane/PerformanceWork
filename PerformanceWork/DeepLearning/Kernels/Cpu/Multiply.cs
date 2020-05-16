using PerformanceWork.OptimizedNumerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.DeepLearning.Kernels.Cpu
{
    public unsafe partial class CpuKernels
    {

        public static Tensor MultiplyFloat(Tensor a, Tensor b)
        {
            Tensor res = new Tensor(a.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWiseMultiplyAVX((float*)a.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

        public static Tensor MultiplyFloat_GetGradient_0(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(s.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)b.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }

        public static Tensor MultiplyFloat_GetGradient_1(Tensor s, Tensor a, Tensor b)
        {
            Tensor res = new Tensor(s.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWiseMultiplyAVX((float*)s.Array, (float*)a.Array, (float*)res.Array, res.Shape.TotalSize);
            return res;
        }
    }
}
