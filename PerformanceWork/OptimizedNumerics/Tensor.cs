using PerformanceWork.OptimizedNumerics.Pool;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics
{
    public unsafe class Tensor<T> : IDisposable where T : struct
    {
        public static ArrayPool<T> Host = ArrayPool<T>.Create(30000000, 300000);
        public static List<ArrayPool<T>> GPU = new List<ArrayPool<T>>();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool<T> GetDevicePool(int deviceId)
        {
            for (int i = GPU.Count; i <= deviceId; i++)
                GPU.Add(ArrayPool<T>.Create(30000000, 300000, true, i));
            return GPU[deviceId];
        }

        public bool OnGPU { get; set; }
        public int DeviceID { get; set; }

        public Shape Shape { get; private set; }

        public void* Array;
        private int LengthOfArray;
        private bool ArrayReturned;
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor(Shape s, bool onGPU = false, int deviceId = -1) //the fastest way
        {
            Initialize(s, onGPU, deviceId);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2) a, bool onGPU = false, int deviceId = -1) //slow way to create tensor because of shape
        {
            Initialize(new Shape(a.d1, a.d2), onGPU, deviceId);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3) a, bool onGPU = false, int deviceId = -1)
        {
            Initialize(new Shape(a.d1, a.d2, a.d3), onGPU, deviceId);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4) a, bool onGPU = false, int deviceId = -1)
        {
            Initialize(new Shape(a.d1, a.d2, a.d3, a.d4), onGPU, deviceId);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Tensor((int d1, int d2, int d3, int d4, int d5) a, bool onGPU = false, int deviceId = -1)
        {
            Initialize(new Shape(a.d1, a.d2, a.d3, a.d4, a.d5), onGPU, deviceId);
        }      

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void Initialize(Shape s, bool onGPU, int deviceId)
        {
            this.Shape = s;
            this.OnGPU = onGPU;
            this.DeviceID = deviceId;

            if (OnGPU)
                Array = GetDevicePool(this.DeviceID).Rent(Shape.Multiplied[0], out LengthOfArray);
            else
                Array = Host.Rent(Shape.Multiplied[0], out LengthOfArray);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetValue(float value)
        {
            Vectorization.ElementWiseSetValueAVX((float*)Array, value, Shape.Multiplied[0]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetValue(Tensor<float> value)
        {
            Vectorization.ElementWiseAssignAVX((float*)Array, (float*)value.Array, Shape.Multiplied[0]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MakeNegative()
        {
            Vectorization.MakeNegativeAVX((float*)Array, (float*)Array, Shape.Multiplied[0]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Add(Tensor<float> m)
        {
            Vectorization.ElementWiseAddAVX((float*)this.Array, (float*)m.Array, (float*)this.Array, this.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void MultiplyBy(float x)
        {
            Vectorization.ElementWiseMultiplyAVX((float*)this.Array, x, (float*)this.Array, this.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void DivideBy(float x)
        {
            Vectorization.ElementWiseMultiplyAVX((float*)this.Array, 1 / x, (float*)this.Array, this.Shape.TotalSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor<float> Clone(Tensor<float> m)
        {
            Tensor<float> n = new Tensor<float>(m.Shape.Clone(), m.OnGPU, m.DeviceID);
            Vectorization.ElementWiseAssignAVX((float*)n.Array, (float*)m.Array, n.Shape.TotalSize);
            return n;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            if (ArrayReturned)
                throw new Exception("The Tensor is already disposed!");

            ArrayReturned = true;

            Shape.Return(this.Shape);

            if (OnGPU)
                GPU[DeviceID].Return(Array, LengthOfArray);
            else
                Host.Return(Array, LengthOfArray);
            GC.SuppressFinalize(this);
        }

        public override string ToString()
        {
            StringBuilder a = new StringBuilder();
            float* ptr = (float*)Array;
            for (int i = 0; i < Shape.TotalSize; i++)
                a.Append(ptr[i] + ", ");
            string res = a.ToString();
            a.Clear();
            return res;
        }
        #region Static Methods

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static bool DeviceCompatibilityCheck(Tensor<T> t1, Tensor<T> t2)
        {
            return t1.OnGPU == t2.OnGPU && (t1.OnGPU == false || t1.DeviceID == t2.DeviceID);
        }

        /// <summary>
        /// Creates an Identity Tensor using the double of the shape.
        /// </summary>
        /// <param name="s">The shape of the identity tensor.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe static Tensor<T> DerivativeIdentity(Shape s)
        {
            Tensor<T> t = new Tensor<T>(s.Clone());
            if(typeof(T) == typeof(float))
            {
                t.SetValue(1.0f);
            }
            else
                throw new Exception("Unsupported number type!");

            return t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor<float> MatrixMultiply(Tensor<float> a, Tensor<float> b)
        {
            if (!Tensor<float>.DeviceCompatibilityCheck(a, b))
                throw new Exception("Tensors are allocted on different devices!");

#if ShapeCheck

            if (a.Shape.N != 2 || b.Shape.N != 2 || a.Shape[1] != b.Shape[0])
                throw new Exception("Tensors are not suitable for matmul!");
#endif
            Shape sc = Shape.NewShape(a.Shape[0], b.Shape[1]);
            Tensor<float> c = new Tensor<float>(sc, a.OnGPU, a.DeviceID);
            Vectorization.MatrixMultiply(a, b, c);
            return c;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor<T> Sum(Tensor<T> t1, Tensor<T> t2)
        {
            if (!Tensor<T>.DeviceCompatibilityCheck(t1, t2))
                throw new Exception("Tensors are allocted on different devices!");
            if (t1.Shape.TotalSize != t2.Shape.TotalSize)
                throw new Exception("Size of Tensors are different!");

            Tensor<T> res = new Tensor<T>(t1.Shape.Clone(), t1.OnGPU, t1.DeviceID);

            if (t1.OnGPU)
            {
                if (typeof(T) == typeof(float))
                {
                    throw new Exception("Unsupported number type!");
                }
                else
                    throw new Exception("Unsupported number type!");
            }
            else
            {
                if (typeof(T) == typeof(float))
                {
                    Vectorization.ElementWiseAddAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported number type!");

            }

            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Sum(Tensor<T> t1, Tensor<T> t2, Tensor<T> res)
        {
            if (!Tensor<T>.DeviceCompatibilityCheck(t1, t2))
                throw new Exception("Tensors are allocted on different devices!");
            if (!Tensor<T>.DeviceCompatibilityCheck(t1, res))
                throw new Exception("Tensors are allocted on different devices!");


            if (t1.OnGPU)
            {
                if (typeof(T) == typeof(float))
                {
                    throw new Exception("Unsupported number type!");
                }
                else
                    throw new Exception("Unsupported number type!");
            }
            else
            {
                if (typeof(T) == typeof(float))
                {
                    Vectorization.ElementWiseAddAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported number type!");
            }
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Tensor<T> Subtract(Tensor<T> t1, Tensor<T> t2)
        {
            if (!Tensor<T>.DeviceCompatibilityCheck(t1, t2))
                throw new Exception("Tensors are allocted on different devices!");
            if (t1.Shape.TotalSize != t2.Shape.TotalSize)
                throw new Exception("Size of Tensors are different!");

            Tensor<T> res = new Tensor<T>(t1.Shape.Clone(), t1.OnGPU, t1.DeviceID);

            if (t1.OnGPU)
            {
                if (typeof(T) == typeof(float))
                {
                    throw new Exception("Unsupported number type!");
                }
                else
                    throw new Exception("Unsupported number type!");
            }
            else
            {
                if (typeof(T) == typeof(float))
                {
                    Vectorization.ElementWiseSubtractAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported number type!");

            }

            return res;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Subtract(Tensor<T> t1, Tensor<T> t2, Tensor<T> res)
        {
            if (!Tensor<T>.DeviceCompatibilityCheck(t1, t2))
                throw new Exception("Tensors are allocted on different devices!");
            if (!Tensor<T>.DeviceCompatibilityCheck(t1, res))
                throw new Exception("Tensors are allocted on different devices!");


            if (t1.OnGPU)
            {
                if (typeof(T) == typeof(float))
                {
                    throw new Exception("Unsupported number type!");
                }
                else
                    throw new Exception("Unsupported number type!");
            }
            else
            {
                if (typeof(T) == typeof(float))
                {
                    Vectorization.ElementWiseSubtractAVX((float*)t1.Array, (float*)t2.Array, (float*)res.Array, t1.Shape.TotalSize);
                }
                else
                    throw new Exception("Unsupported number type!");
            }
        }
        
        #endregion

    }
}
