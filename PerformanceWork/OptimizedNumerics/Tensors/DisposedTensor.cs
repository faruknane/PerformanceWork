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
    internal unsafe class DisposedTensor : Tensor
    {
        public static int TEMP = 0;

        public GCHandle Handle;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private DisposedTensor(Shape s, GCHandle handle, NumberType type) : base(s, (void*)handle.AddrOfPinnedObject(), new TensorConfig(Device.Host, type))
        {
            this.Handle = handle;
        }

        public static DisposedTensor Create(Array arr, Shape s, NumberType type)
        {
            return new DisposedTensor(s, GCHandle.Alloc(arr, GCHandleType.Pinned), type);
        }
       
        public void UnpinPointer()
        {
            Handle.Free();
        }

        ~DisposedTensor()
        {
            UnpinPointer();
            Interlocked.Increment(ref TEMP);
        }
    }
}
