using PerformanceWork.NCuda;
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
        public TensorConfig Config;

        public void* Array;
        public bool ArrayReturned;


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Tensor(Shape s, void* ptr, TensorConfig devconfig)
        {
            this.Shape = s;
            this.Array = ptr;
            this.Config = devconfig;
            ArrayReturned = true; //means that the array wont be returned to the pool.
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor(Shape s, TensorConfig devconfig)
        {
            Initialize(s, devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2) a, TensorConfig devconfig)
        {
            Initialize(new Shape(a.d1, a.d2), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3) a, TensorConfig devconfig)
        {
            Initialize(new Shape(a.d1, a.d2, a.d3), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4) a, TensorConfig devconfig)
        {
            Initialize(new Shape(a.d1, a.d2, a.d3, a.d4), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4, int d5) a, TensorConfig devconfig)
        {
            Initialize(new Shape(a.d1, a.d2, a.d3, a.d4, a.d5), devconfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void Initialize(Shape s, TensorConfig devconfig)
        {
            this.Shape = s;
            this.Config = devconfig;

            Array = TensorPool.GetDevicePool(this.Config.Device).Rent(Shape.Multiplied[0], this.Config.GetUnitLength());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetFloat(float value)
        {
            if (this.Config == TensorConfig.Host_Float32)
                VectorizationFloat.ElementWiseSetValueAVX((float*)Array, value, Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetTensor(Tensor value)
        {
            if (this.Config != value.Config)
                throw new Exception("Device Configuration is different!");

            if (this.Config == TensorConfig.Host_Float32)
                VectorizationFloat.ElementWiseAssignAVX((float*)Array, (float*)value.Array, Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MakeNegative()
        {
            if (this.Config == TensorConfig.Host_Float32)
                VectorizationFloat.MakeNegativeAVX((float*)Array, (float*)Array, Shape.Multiplied[0]);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void AddTensor(Tensor value)
        {
            if (this.Config != value.Config)
                throw new Exception("Device Configuration is different!");

            if (this.Config == TensorConfig.Host_Float32)
                    VectorizationFloat.ElementWiseAddAVX((float*)this.Array, (float*)value.Array, (float*)this.Array, this.Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MultiplyByFloat(float x)
        {
            if (this.Config == TensorConfig.Host_Float32)
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)this.Array, x, (float*)this.Array, this.Shape.TotalSize);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void DivideByFloat(float x)
        {
            if (this.Config == TensorConfig.Host_Float32)
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)this.Array, 1 / x, (float*)this.Array, this.Shape.TotalSize);
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
                TensorPool.GetDevicePool(this.Config.Device).Return(Array, Shape.TotalSize, this.Config.GetUnitLength());
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

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override string ToString()
        {
            static string CPUTensorFloat32ToString(Tensor x)
            {
                StringBuilder a = new StringBuilder();
                float* ptr = (float*)x.Array;

                if (x.Shape.N == 2)
                {
                    a.Append("[");
                    for (int i = 0; i < x.Shape[0]; i++)
                    {
                        a.Append("[");
                        for (int j = 0; j < x.Shape[1]; j++)
                            a.Append($"{ptr[x.Shape.Index(i, j)]}" + (j == x.Shape[1] - 1 ? "" : ", "));
                        a.Append("]");
                    }
                    a.Append("]");
                }
                else
                {
                    a.Append("[");
                    for (int i = 0; i < x.Shape[0]; i++)
                        a.Append($"{ptr[x.Shape.Index(i)]}" + (i == x.Shape[0] - 1 ? "" : ", "));
                    a.Append("]");
                }

                string res = a.ToString();
                a.Clear();

                return res;
            }
            if (this.Config == TensorConfig.Host_Float32)
                return CPUTensorFloat32ToString(this);
            else if(this.Config == TensorConfig.NvidiaGPU_Float32)
            {
                Tensor m = Tensor.CopyTo(this, Device.Host);
                string str = CPUTensorFloat32ToString(m);
                m.Dispose();
                return str;
            }
            else
                throw new Exception("Unsupported Tensor Configuration!");
        }

        public Tensor CopyTo(Device dev)
        {
            return Tensor.CopyTo(this, dev);
        }
        public Tensor Clone()
        {
            return Tensor.Clone(this);
        }

        #region Static Methods

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Clone(Tensor m)
        {
            return Tensor.CopyTo(m, m.Config.Device);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor CopyTo(Tensor m, Device dev)
        {
            TensorConfig conf = new TensorConfig(dev, m.Config.NumType);
            Tensor t = new Tensor(m.Shape.Clone(), conf);
            if(m.Config.Device.Type == DeviceType.Host && dev.Type == DeviceType.Host)
            {
                VectorizationFloat.ElementWiseAssignAVX((float*)t.Array, (float*)m.Array, t.Shape.TotalSize);
            }
            else if (m.Config.Device.Type == DeviceType.Host && dev.Type == DeviceType.NvidiaGPU)
            {
                CudaManagement.CopyArray(m.Array, t.Array, m.Shape.TotalSize * m.Config.GetUnitLength());
            }
            else if (m.Config.Device.Type == DeviceType.NvidiaGPU && dev.Type == DeviceType.Host)
            {
                CudaManagement.CopyArray(m.Array, t.Array, m.Shape.TotalSize * m.Config.GetUnitLength());
            }
            else if (m.Config.Device.Type == DeviceType.NvidiaGPU && dev.Type == DeviceType.NvidiaGPU)
            {
                CudaManagement.CopyArray(m.Array, t.Array, m.Shape.TotalSize * m.Config.GetUnitLength());
            }
            return t;
        }


        /// <summary>
        /// Creates an Identity Tensor using the double of the shape.
        /// </summary>
        /// <param name="s">The shape of the identity tensor.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static Tensor DerivativeIdentity(Shape s, TensorConfig deviceConfig)
        {
            Tensor t = new Tensor(s.Clone(), deviceConfig);
            if (t.Config == TensorConfig.Host_Float32)
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

            if (data.Config == TensorConfig.Host_Float32)
                return new Tensor(s, (void*)((long)(data.Array) + begin * data.Config.GetUnitLength()), data.Config);
            else
                throw new Exception("Unsupported Device Configuration!");
        }

        /// <summary>
        ///  Creates a tensor on Host Device. It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="s">Shape for the tensor to be created.</param>
        /// <param name="type">NumberType for the tensor to be created.</param>
        /// <returns></returns>
        public static unsafe Tensor ToDisposedTensor(Array data, Shape s, NumberType type)
        {
            if (data.LongLength != s.TotalSize)
                throw new Exception("Cant convert data into the given shape");

            if (type == NumberType.Float32)
            {
                if (data is float[])
                    fixed (float* ptr = (float[])data)
                        return new Tensor(s, ptr, new TensorConfig(Device.Host, type));
                else if (data is float[,])
                    fixed (float* ptr = (float[,])data)
                        return new Tensor(s, ptr, new TensorConfig(Device.Host, type));
                else if (data is float[,,])
                    fixed (float* ptr = (float[,,])data)
                        return new Tensor(s, ptr, new TensorConfig(Device.Host, type));
                else if (data is float[,,,])
                    fixed (float* ptr = (float[,,,])data)
                        return new Tensor(s, ptr, new TensorConfig(Device.Host, type));
                else if (data is float[,,,,])
                    fixed (float* ptr = (float[,,,,])data)
                        return new Tensor(s, ptr, new TensorConfig(Device.Host, type));
                else if (data is float[,,,,,])
                    fixed (float* ptr = (float[,,,,,])data)
                        return new Tensor(s, ptr, new TensorConfig(Device.Host, type));
                else
                    throw new Exception("Unsupported Array Type");
            }
            else
                throw new Exception("Unsupported NumberType!");
        }

        
        public static unsafe Tensor NewDisposedTensor(Shape s, void* ptr, TensorConfig tensorConfig)
        {
            return new Tensor(s, ptr, tensorConfig);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            if (a.Shape.N != 2 || b.Shape.N != 2 || a.Shape[1] != b.Shape[0])
                throw new Exception("Shape error!");

            if (a.Config != b.Config)
                throw new Exception("Device Configurations are different!");

            if (a.Config == TensorConfig.Host_Float32)
            {
                Shape sc = new Shape(a.Shape[0], b.Shape[1]);
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

            if (t1.Config == TensorConfig.Host_Float32)
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

            if (t1.Config == TensorConfig.Host_Float32)
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
