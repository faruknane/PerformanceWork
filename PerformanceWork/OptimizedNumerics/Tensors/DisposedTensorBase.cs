using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics.Tensors
{
    public unsafe class DisposedTensorBase : TensorBase
    {
        public static int TEMP = 0;

        public GCHandle Handle;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public DisposedTensorBase(GCHandle handle, long length, bool returned, TensorConfig conf) : base((void*)handle.AddrOfPinnedObject(), length, returned, conf)
        {
            this.Handle = handle;
        }
       
        public void UnpinPointer()
        {
            Handle.Free();
        }

        ~DisposedTensorBase()
        {
            UnpinPointer();
            Interlocked.Increment(ref TEMP);
        }
    }
}
