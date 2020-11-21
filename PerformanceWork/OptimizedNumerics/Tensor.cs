using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Tensor : IDisposable
    {
        public Shape Shape { get; private set; }
        public DeviceConfig Config;

        public void* Array;
        public bool ArrayReturned;


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Tensor(Shape s, void* ptr, DeviceConfig devconfig)
        {
            this.Shape = s;
            this.Array = ptr;
            this.Config = devconfig;
            ArrayReturned = true; //means that the array wont be returned to the pool.
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor(Shape s, DeviceConfig devconfig)
        {
            Initialize(s, devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2) a, DeviceConfig devconfig)
        {
            Initialize(new Shape((a.d1, a.d2)), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3) a, DeviceConfig devconfig)
        {
            Initialize(new Shape((a.d1, a.d2, a.d3)), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4) a, DeviceConfig devconfig)
        {
            Initialize(new Shape((a.d1, a.d2, a.d3, a.d4)), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4, int d5) a, DeviceConfig devconfig)
        {
            Initialize(new Shape((a.d1, a.d2, a.d3, a.d4, a.d5)), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void Initialize(Shape s, DeviceConfig devconfig)
        {
            this.Shape = s;
            this.Config = devconfig;

            Array = TensorPool.GetDevicePool(this.Config).Rent(Shape.Multiplied[0], this.Config.GetUnitLength());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetFloat(float value)
        {
            if (this.Config == DeviceConfig.Host_Float)
                VectorizationFloat.ElementWiseSetValueAVX((float*)Array, value, Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetTensor(Tensor value)
        {
            if (this.Config != value.Config)
                throw new Exception("Device Configuration is different!");

            if (this.Config == DeviceConfig.Host_Float)
                VectorizationFloat.ElementWiseAssignAVX((float*)Array, (float*)value.Array, Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MakeNegative()
        {
            if (this.Config == DeviceConfig.Host_Float)
                VectorizationFloat.MakeNegativeAVX((float*)Array, (float*)Array, Shape.Multiplied[0]);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void AddTensor(Tensor value)
        {
            if (this.Config != value.Config)
                throw new Exception("Device Configuration is different!");

            if (this.Config == DeviceConfig.Host_Float)
                    VectorizationFloat.ElementWiseAddAVX((float*)this.Array, (float*)value.Array, (float*)this.Array, this.Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MultiplyByFloat(float x)
        {
            if (this.Config == DeviceConfig.Host_Float)
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)this.Array, x, (float*)this.Array, this.Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void DivideByFloat(float x)
        {
            if (this.Config == DeviceConfig.Host_Float)
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)this.Array, 1 / x, (float*)this.Array, this.Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Clone(Tensor m)
        {
            if (m.Config == DeviceConfig.Host_Float)
            {
                Tensor n = new Tensor(m.Shape.Clone(), m.Config);
                VectorizationFloat.ElementWiseAssignAVX((float*)n.Array, (float*)m.Array, n.Shape.TotalSize);
                return n;
            }
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            Dispose(false);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
                TensorPool.GetDevicePool(this.Config).Return(Array, Shape.TotalSize);
                if (!gc)
                    GC.SuppressFinalize(this);
                DisposedCount++;
            }
        }

        public static int DisposedCount = 0;

        ~Tensor()
        {
            Dispose(true);
        }

        public override string ToString()
        {
            if (this.Config == DeviceConfig.Host_Float)
            {
                StringBuilder a = new StringBuilder();
                float* ptr = (float*)Array;

                if (this.Shape.N == 2)
                {
                    a.Append("[");
                    for (int i = 0; i < Shape[0]; i++)
                    {
                        a.Append("[");
                        for (int j = 0; j < Shape[1]; j++)
                            a.Append($"{ptr[Shape.Index(i, j)]}" + (j == Shape[1] - 1 ? "" : ", "));
                        a.Append("]");
                    }
                    a.Append("]");
                }
                else
                {
                    for (int i = 0; i < Shape.TotalSize; i++)
                        a.Append(ptr[i] + ", ");
                }

                string res = a.ToString();
                a.Clear();

                return res;
            }
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        #region Static Methods

        /// <summary>
        /// Creates an Identity Tensor using the double of the shape.
        /// </summary>
        /// <param name="s">The shape of the identity tensor.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static Tensor DerivativeIdentity(Shape s, DeviceConfig deviceConfig)
        {
            Tensor t = new Tensor(s.Clone(), deviceConfig);
            if (t.Config == DeviceConfig.Host_Float)
                t.SetFloat(1.0f);
            else
                throw new Exception("Unsupported Device Configuration!");
            return t;
        }

        //todo create unit test for Cut
        /// <summary>
        /// Slices the tensor data between begin and end indices into the shape s. It assumes that the sliced tensor is already returned. So, It won't do anything if the sliced tensor gets disposed.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="begin"></param>
        /// <param name="end"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public static unsafe Tensor Cut(Tensor data, int begin, Shape s)
        {
            if (data.Shape.TotalSize < begin + s.TotalSize)
                throw new Exception("data.Shape.TotalSize < begin + s.TotalSize!");

            return new Tensor(s, (void*)((long)(data.Array) + begin * data.Config.GetUnitLength()), data.Config);
        }

        /// <summary>
        ///  It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="s"></param>
        /// <param name="dev"></param>
        /// <returns></returns>
        public static unsafe Tensor LoadArrayToDisposedTensor(Array data, Shape s, DeviceConfig dev)
        {
            if (data.LongLength != s.TotalSize)
                throw new Exception("Cant convert data into the given shape");

            if (dev.NumType == DeviceConfig.NumberType.CPUFloat)
            {
                if (data is float[])
                    fixed (float* ptr = (float[])data)
                        return new Tensor(s, ptr, dev);
                else if (data is float[,])
                    fixed (float* ptr = (float[,])data)
                        return new Tensor(s, ptr, dev);
                else if (data is float[,,])
                    fixed (float* ptr = (float[,,])data)
                        return new Tensor(s, ptr, dev);
                else if (data is float[,,,])
                    fixed (float* ptr = (float[,,,])data)
                        return new Tensor(s, ptr, dev);
                else if (data is float[,,,,])
                    fixed (float* ptr = (float[,,,,])data)
                        return new Tensor(s, ptr, dev);
                else if (data is float[,,,,,])
                    fixed (float* ptr = (float[,,,,,])data)
                        return new Tensor(s, ptr, dev);
                else
                    throw new Exception("Unsupported Array Type");
            }
            else
                throw new Exception("Unsupported Device Configuration!");
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            if (a.Shape.N != 2 || b.Shape.N != 2 || a.Shape[1] != b.Shape[0])
                throw new Exception("Shape error!");

            if (a.Config != b.Config)
                throw new Exception("Device Configurations are different!");

            if (a.Config == DeviceConfig.Host_Float)
            {
                Shape sc = new Shape((a.Shape[0], b.Shape[1]));
                Tensor c = new Tensor(sc, a.Config);
                VectorizationFloat.MatrixMultiply(a, b, c);
                return c;
            }
            else
                throw new Exception("Unsupported Device Configuration!");

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Sum(Tensor t1, Tensor t2)
        {
            if (t1.Config != t2.Config)
                throw new Exception("Device Configurations are different!");

            if (t1.Shape.TotalSize != t2.Shape.TotalSize)
                throw new Exception("Size of Tensors are different!");

            Tensor res = new Tensor(t1.Shape.Clone(), t1.Config);

            if (t1.Config == DeviceConfig.Host_Float)
            {
                VectorizationFloat.ElementWiseAddAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
            }
            else
                throw new Exception("Unsupported Device Configuration!");

            return res;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Subtract(Tensor t1, Tensor t2)
        {
            if (t1.Config != t2.Config)
                throw new Exception("Device Configurations are different!");

            if (t1.Shape.TotalSize != t2.Shape.TotalSize)
                throw new Exception("Size of Tensors are different!");

            Tensor res = new Tensor(t1.Shape.Clone(), t1.Config);

            if (t1.Config == DeviceConfig.Host_Float)
            {
                VectorizationFloat.ElementWiseSubtractAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
            }
            else
                throw new Exception("Unsupported Device Configuration!");

            return res;
        }

        #endregion

    }
}
