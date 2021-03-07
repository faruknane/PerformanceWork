using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics
{
    public static class TensorExtension
    {

        /// <summary>
        ///  It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="s"></param>
        /// <param name="dev"></param>
        /// <returns></returns>
        public static unsafe Tensor ToDisposedTensor(this Array data, Shape s, NumberType type)
        {
            return Tensor.ToDisposedTensor(data, s, type);
        }

    }
}
