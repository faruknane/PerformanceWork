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
        public static Tensor SigmoidFloat(Tensor v)
        {
            Tensor sigmo = new Tensor(v.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.Sigmoid((float*)v.Array, (float*)sigmo.Array, sigmo.Shape.TotalSize);
            return sigmo;
        }

        public static Tensor SigmoidFloat_GetGradient_0(Tensor s, Tensor sigmo)
        {
            Tensor combined = new Tensor(s.Shape.Clone(), DataType.Type.Float, DeviceIndicator.Host());
            VectorizationFloat.ElementWise_A_MultipliedBy_1_Minus_A_MultipliedByB((float*)sigmo.Array, (float*)s.Array, (float*)combined.Array, sigmo.Shape.TotalSize);
            return combined;
        }

    }
}
