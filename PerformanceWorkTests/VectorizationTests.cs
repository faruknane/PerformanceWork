using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork.OptimizedNumerics;
using System;

namespace PerformanceWorkTests
{
    [TestClass]
    public class VectorizationTests
    {
        [TestMethod]
        public void Add()
        {
            float[] v1 = { 1, 2, 3 };
            float[] v2 = { 1, 2, 3 };
            float[] res = new float[3];
            Vectorization.AddAVX(ref v1, ref v2, ref res, res.Length);
            float[] res2 = { 2, 4, 6 };

            Assert.IsTrue(ArrayEqual<float>(res, res2));
        }
        [TestMethod]
        public void DotProduct()
        {
            int size = 101;
            float[] v1 = new float[size];
            float[] v2 = new float[size];
            for (int i = 0; i < size; i++)
                v1[i] = v2[i] = i;

            double res = Vectorization.DotProductAVX(ref v1, ref v2, size);

            double res2 = 0;
            for (int i = 0; i < size; i++)
                res2 += v1[i] * v2[i];
            Assert.AreEqual(res, res2);
        }

        [TestMethod]
        public unsafe void DotProductPointer()
        {
            int size = 101;
            float[] v1 = new float[size];
            float[] v2 = new float[size];
            for (int i = 0; i < size; i++)
                v1[i] = v2[i] = i;
            fixed (float* ptr = v1, ptr2 = v2)
            {
                double res = Vectorization.DotProductAVX(ptr, ptr2, size);
                double res2 = 0;
                for (int i = 0; i < size; i++)
                    res2 += v1[i] * v2[i];
                Assert.AreEqual(res, res2);
            }
        }

        [TestMethod]
        public unsafe void DotProductParallel()
        {
            for(long size = 1; size < 300; size++)
            {
                float[] v1 = new float[size];
                float[] v2 = new float[size];

                for (int i = 0; i < size; i++)
                    v1[i] = v2[i] = i;

                fixed (float* ptr = v1, ptr2 = v2)
                {
                    double res = Vectorization.DotProductAVXParallel(ptr, ptr2, (int)size);
                    long res2 = size * (size - 1) * (2 * size - 1) / 6;
                    Assert.AreEqual(res, (double)res2);
                }
            }
            
        }

        [TestMethod]
        public void Multiply()
        {
            float[] v1 = { 1, 2, 3 };
            float[] v2 = { 1, 2, 3 };
            float[] res = new float[3];
            Vectorization.MultiplyAVX(ref v1, ref v2, ref res, res.Length);
            float[] res2 = { 1, 4, 9 };

            Assert.IsTrue(ArrayEqual<float>(res, res2));
        }

        [TestMethod]
        public void MatrixMultiplyTranspose()
        {
            int x = 50, y = 100, z = 30;
            Matrix a = new Matrix(x, y);
            Matrix b = new Matrix(y, z);
            Matrix res = new Matrix(x, z);
            Matrix res2 = new Matrix(x, z);

            Random r = new Random();

            


        }


        public bool ArrayEqual<T>(T[] v1, T[] v2)
        {
            if (v1.Length != v2.Length) return false;
            for (int i = 0; i < v1.Length; i++)
                if (!v1[i].Equals(v2[i]))
                    return false;
            return true;
        }
    }
}
