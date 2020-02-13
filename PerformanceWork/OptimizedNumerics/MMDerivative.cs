//using PerformanceWork.OptimizedNumerics.Pool;
//using System;
//using System.Collections.Generic;
//using System.Runtime.CompilerServices;
//using System.Text;

//namespace PerformanceWork.OptimizedNumerics
//{
//    public unsafe class MMDerivative
//    {
//        public int D1 { get; set; }
//        public int D2 { get; set; }
//        public int D3 { get; set; }
//        public int D4 { get; set; }
//        public bool Negative { get; set; } = false;

//        public float* Derivatives;
//        private int length;
//        public static ArrayPool<float> Pool2 = ArrayPool<float>.Create(1000000, 100000);

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public MMDerivative(int d1, int d2, int d3, int d4, bool setzero)
//        {
//            D1 = d1;
//            D2 = d2;
//            D3 = d3;
//            D4 = d4;

//            Derivatives = (float*)Pool2.Rent(d1 * d2 * d3 * d4, out length);
           
//            if (setzero)
//            {
//                Vectorization.ElementWiseSetValueAVX(Derivatives, 0, d1 * d2 * d3 * d4);
//            }
//        }

//        int x = 0;
//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void Dispose()
//        {
//            if (x > 0)
//            {
//                Console.WriteLine("array is returned already");
//            }
//            GC.SuppressFinalize(this);
//            Pool2.Return(Derivatives, length);
//            x++;
//        }

//        public float this[int x1, int x2, int x3, int x4]
//        {
//            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//            get
//            {
//                return Derivatives[x1 * D2 * D3 * D4 + x2 * D3 * D4 + x3 * D4 + x4];
//            }
//            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//            set
//            {
//                Derivatives[x1 * D2 * D3 * D4 + x2 * D3 * D4 + x3 * D4 + x4] = value;
//            }
//        }

//        public static MMDerivative Identitiy(int d1, int d2)
//        {
//            MMDerivative res = new MMDerivative(d1, d2, d1, d2, true);
//            for (int i = 0; i < d1; i++)
//                for (int i2 = 0; i2 < d2; i2++)
//                    res[i, i2, i, i2] = 1;
//            return res;
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void MultiplyBy(float x)
//        {
//            Vectorization.ElementWiseMultiplyAVX(this.Derivatives, x, this.Derivatives, D1 * D2 * D3 * D4);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void Add(MMDerivative m)
//        {
//            Vectorization.ElementWiseAddAVX(this.Derivatives, m.Derivatives, this.Derivatives, D1 * D2 * D3 * D4);
//        }

//        public static MMDerivative Clone(MMDerivative m)
//        {
//            MMDerivative n = new MMDerivative(m.D1, m.D2, m.D3, m.D4, false);
//            n.Negative = m.Negative;
//            Vectorization.ElementWiseAssignAVX(n.Derivatives, m.Derivatives, m.D1 * m.D2 * m.D3 * m.D4);
//            return n;
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void DivideBy(float x)
//        {
//            Vectorization.ElementWiseMultiplyAVX(this.Derivatives, 1 / x, this.Derivatives, D1 * D2 * D3 * D4);
//        }
//    }
//}
