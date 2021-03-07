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
        private DisposedTensor(Shape s, void* ptr, NumberType type) : base(s, ptr, new TensorConfig(Device.Host, type))
        {

        }

        public static DisposedTensor Create(Array arr, Shape s, NumberType type)
        {
            GCHandle h = GCHandle.Alloc(arr, GCHandleType.Pinned);
            void* ptr = (void*)h.AddrOfPinnedObject();
            DisposedTensor d = new DisposedTensor(s, ptr, type);
            d.Handle = h;
            return d;
        }

       
        public void UnPinPointer()
        {
            Handle.Free();
        }

        ~DisposedTensor()
        {
            UnPinPointer();
            Interlocked.Increment(ref TEMP);
        }
    }
}
