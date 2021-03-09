using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics.Tensors
{
    public unsafe class TensorBase : IDisposable
    {
        public static int DisposedCount = 0;

        public void* Array;
        public bool ArrayReturned;
        public long Length;
        public TensorConfig Config;

        public TensorBase(void* ptr, long length, bool returned, TensorConfig conf)
        {
            this.Array = ptr;
            this.Length = length;
            this.ArrayReturned = returned;
            this.Config = conf;
        }

        public TensorBase(long length, TensorConfig conf)
        {
            this.Length = length;
            this.ArrayReturned = false;
            this.Config = conf;
            var ptr = TensorPool.GetDevicePool(this.Config.Device).Rent(length, this.Config.GetUnitLength());
            this.Array = ptr;

        }

        public void Dispose(bool gc)
        {
            if (!gc && ArrayReturned)
            {
                System.Diagnostics.StackTrace t = new System.Diagnostics.StackTrace();
                Console.WriteLine("StackTrace: '{0}'", Environment.StackTrace);
                throw new Exception("The tensor is already disposed!");
            }

            if (!ArrayReturned)
            {
                ArrayReturned = true;
                TensorPool.GetDevicePool(this.Config.Device).Return(Array, Length, this.Config.GetUnitLength());
                if (!gc)
                    GC.SuppressFinalize(this);
                Interlocked.Increment(ref TensorBase.DisposedCount);
            }
        }

        public void Dispose()
        {
            Dispose(false);
        }

        ~TensorBase()
        {
            Dispose(true);
        }
    }
}
