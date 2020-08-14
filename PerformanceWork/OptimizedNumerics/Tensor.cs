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
        public DataType.Type Type;
        public DeviceIndicator Device;

        public void* Array;
        private int LengthOfArray;
        public bool ArrayReturned;


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Tensor(Shape s, void* ptr, DataType.Type type, DeviceIndicator device)
        {
            this.Shape = s;
            this.Array = ptr;
            this.Type = type;
            this.Device = device;
            ArrayReturned = true; //means that the array wont be returned to the pool.
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor(Shape s, DataType.Type type, DeviceIndicator device)
        {
            Initialize(s, type, device);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2) a, DataType.Type type, DeviceIndicator device)
        {
            Initialize(new Shape((a.d1, a.d2)), type, device);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3) a, DataType.Type type, DeviceIndicator device)
        {
            Initialize(new Shape((a.d1, a.d2, a.d3)), type, device);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4) a, DataType.Type type, DeviceIndicator device)
        {
            Initialize(new Shape((a.d1, a.d2, a.d3, a.d4)), type, device);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4, int d5) a, DataType.Type type, DeviceIndicator device)
        {
            Initialize(new Shape((a.d1, a.d2, a.d3, a.d4, a.d5)), type, device);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void Initialize(Shape s, DataType.Type type, DeviceIndicator device)
        {
            this.Shape = s;
            this.Device = device;
            this.Type = type;

            Array = TensorPool.GetDevicePool(this.Device).Rent(Shape.Multiplied[0], out LengthOfArray, this.Type);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetFloat(float value)
        {
            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseSetValueAVX((float*)Array, value, Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetTensor(Tensor value)
        {
            if (this.Type != value.Type)
                throw new Exception("Data type is different!");

            if (this.Device != value.Device)
                throw new Exception("Allocated memory is on different device!");

            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseAssignAVX((float*)Array, (float*)value.Array, Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MakeNegative()
        {
            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    VectorizationFloat.MakeNegativeAVX((float*)Array, (float*)Array, Shape.Multiplied[0]);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void AddTensor(Tensor value)
        {
            if (this.Type != value.Type)
                throw new Exception("Data type is different!");

            if (this.Device != value.Device)
                throw new Exception("Allocated memory is on different device!");

            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseAddAVX((float*)this.Array, (float*)value.Array, (float*)this.Array, this.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MultiplyByFloat(float x)
        {
            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)this.Array, x, (float*)this.Array, this.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void DivideByFloat(float x)
        {
            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseMultiplyAVX((float*)this.Array, 1 / x, (float*)this.Array, this.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Clone(Tensor m)
        {
            if (m.Device.Type == DeviceType.Host)
            {
                if (m.Type == DataType.Type.Float)
                {
                    Tensor n = new Tensor(m.Shape.Clone(), m.Type, m.Device);
                    VectorizationFloat.ElementWiseAssignAVX((float*)n.Array, (float*)m.Array, n.Shape.TotalSize);
                    return n;
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");
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
                TensorPool.GetDevicePool(this.Device).Return(Array, LengthOfArray);
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
            if (this.Device.Type == DeviceType.Host)
            {
                if (this.Type == DataType.Type.Float)
                {
                    StringBuilder a = new StringBuilder();
                    float* ptr = (float*)Array;

                    if (this.Shape.N == 2)
                    {
                        for (int i = 0; i < Shape[0]; i++)
                        {
                            for (int j = 0; j < Shape[1]; j++)
                                a.Append($"{ptr[Shape.Index(i,j)]}" + (j == Shape[1] - 1 ? "": ", "));
                            a.Append("\n");
                        }
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
                    throw new Exception("Unsupported Data Type!");
            }
            else
                throw new Exception("Unsupported Platform!");
        }

        #region Static Methods

        /// <summary>
        /// Creates an Identity Tensor using the double of the shape.
        /// </summary>
        /// <param name="s">The shape of the identity tensor.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static Tensor DerivativeIdentity(Shape s, DataType.Type dtype, DeviceIndicator device)
        {
            Tensor t = new Tensor(s.Clone(), dtype, device);
            if (t.Type == DataType.Type.Float)
                t.SetFloat(1.0f);
            else 
                throw new Exception("Unsupported data type!");
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
        public static unsafe Tensor Cut(Tensor data, int begin, int end, Shape s)
        {
            if (end - begin != s.TotalSize)
                throw new Exception("Cant convert it into the shape");

            return new Tensor(s, (void*)((long)(data.Array) + begin * DataType.GetByteSize(data.Type)), data.Type, data.Device);
        }

        /// <summary>
        /// Coverts the float array into a float tensor on host. It assumes that the float tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public static unsafe Tensor LoadFloatArray(Array data, Shape s, DeviceIndicator dev)
        {
            if (data.LongLength != s.TotalSize)
                throw new Exception("Cant convert it into the shape");

            if (data.GetType().GetElementType() == typeof(float))
            {
                fixed (float* ptr = (float[])data)
                    return new Tensor(s, (void*)ptr, DataType.Type.Float, dev);
            }
            else
                throw new Exception("unsupported bla bla");
        }

        public static unsafe Tensor LoadFloatArray(Array data, Shape s)
        {
            if (data.LongLength != s.TotalSize)
                throw new Exception("Cant convert it into the shape");

            if (data.GetType().GetElementType() == typeof(float))
            {
                if(data is float[])
                    fixed (float* ptr = (float[])data)
                        return new Tensor(s, (void*)ptr, DataType.Type.Float, DeviceIndicator.Host());
                else if (data is float[,])
                    fixed (float* ptr = (float[,])data)
                        return new Tensor(s, (void*)ptr, DataType.Type.Float, DeviceIndicator.Host());
                else
                    throw new Exception("unsupported bla bla");
            }
            else
                throw new Exception("unsupported bla bla");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            if (a.Type != b.Type)
                throw new Exception("Data type is different!");

            if (a.Device != b.Device)
                throw new Exception("Allocated memory is on different device!");

            if (a.Device.Type == DeviceType.Host)
            {
                if (a.Type == DataType.Type.Float)
                {

                    if (a.Shape.N != 2 || b.Shape.N != 2 || a.Shape[1] != b.Shape[0])
                        throw new Exception("Shape error!");

                    Shape sc = new Shape((a.Shape[0], b.Shape[1]));
                    Tensor c = new Tensor(sc, a.Type, a.Device);
                    VectorizationFloat.MatrixMultiply(a, b, c);
                    return c;
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Sum(Tensor t1, Tensor t2)
        {
            if (t1.Type != t2.Type)
                throw new Exception("Data type is different!");

            if (t1.Device != t2.Device)
                throw new Exception("Allocated memory is on different device!");

            if (t1.Shape.TotalSize != t2.Shape.TotalSize)
                throw new Exception("Size of Tensors are different!");

            Tensor res = new Tensor(t1.Shape.Clone(), t1.Type, t1.Device);

            if (t1.Device.Type == DeviceType.Host)
            {
                if (t1.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseAddAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");

            return res;
        }

      
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor Subtract(Tensor t1, Tensor t2)
        {
            if(t1.Type != t2.Type)
                throw new Exception("Data type is different!");

            if (t1.Device != t2.Device)
                throw new Exception("Allocated memory is on different device!");

            if (t1.Shape.TotalSize != t2.Shape.TotalSize)
                throw new Exception("Size of Tensors are different!");


            Tensor res = new Tensor(t1.Shape.Clone(), t1.Type, t1.Device);

            if (t1.Device.Type == DeviceType.Host)
            {
                if (t1.Type == DataType.Type.Float)
                {
                    VectorizationFloat.ElementWiseSubtractAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported data type!");
            }
            else
                throw new Exception("Unsupported Platform!");

            return res;
        }
       
        #endregion

    }
}
