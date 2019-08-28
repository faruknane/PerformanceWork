using Microsoft.VisualStudio.TestTools.UnitTesting;
using PerformanceWork.OptimizedNumerics;
using System;

namespace PerformanceWorkTests
{
    [TestClass]
    public class VectorizationTests
    {
        [TestMethod]
        public void MatrixSetZero()
        {
            int Size = 123;
            float[] v1 = new float[Size];
            float[] v2 = new float[Size];
            for (int i = 0; i < v1.Length; i++)
                v1[i] = i;

            Vectorization.ElementWiseSetValueAVX(v1, 0, v1.Length);
            Assert.IsTrue(ArrayEqual<float>(v1, v2));
        }

        [TestMethod]
        public void Assigning()
        {
            int Size = 123;
            float[] v1 = new float[Size];
            float[] v2 = new float[Size];
            for (int i = 0; i < v1.Length; i++)
                v1[i] = i;

            Vectorization.ElementWiseAssignAVX(v2, v1, v1.Length);
            Assert.IsTrue(ArrayEqual<float>(v1, v2));
        }

        [TestMethod]
        public void Equality()
        {
            {
                float[] v1 = { 1, 2, 3 };
                float[] v2 = { 1, 2, 3 };
                bool res = Vectorization.ElementWiseEqualsAVX(v1, v2, v1.Length);
                bool res2 = true;
                Assert.AreEqual(res, res2);
            }
            {
                float[] v1 = { 1, 2, 2 };
                float[] v2 = { 1, 2, 3 };
                bool res = Vectorization.ElementWiseEqualsAVX(v1, v2, v1.Length);
                bool res2 = false;
                Assert.AreEqual(res, res2);
            }
        }
        [TestMethod]
        public void Add()
        {
            float[] v1 = { 1, 2, 3 };
            float[] v2 = { 1, 2, 3 };
            float[] res = new float[3];
            Vectorization.ElementWiseAddAVX(v1, v2, res, res.Length);
            float[] res2 = { 2, 4, 6 };

            Assert.IsTrue(ArrayEqual<float>(res, res2));
        }
        [TestMethod]
        public void Add2()
        {
            float[] v1 = { 1, 2, 3 };
            float v2 = 2;
            float[] res = new float[3];
            Vectorization.ElementWiseAddAVX(v1, v2, res, res.Length);
            float[] res2 = { 3, 4, 5 };

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

            double res = Vectorization.DotProductFMA(v1, v2, size);

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
                double res = Vectorization.DotProductFMA(ptr, ptr2, size);
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
                    double res = Vectorization.DotProductFMAParallel(ptr, ptr2, (int)size);
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
            Vectorization.ElementWiseMultiplyAVX(v1, v2, res, res.Length);
            float[] res2 = { 1, 4, 9 };

            Assert.IsTrue(ArrayEqual<float>(res, res2));
        }
        [TestMethod]
        public void Multiply2()
        {
            float[] v1 = { 1, 2, 3 };
            float v2 = 3;
            float[] res = new float[3];
            Vectorization.ElementWiseMultiplyAVX(v1, v2, res, res.Length);
            float[] res2 = { 3, 6, 9 };

            Assert.IsTrue(ArrayEqual<float>(res, res2));
        }

        [TestMethod]
        public void MatrixMultiply()
        {
            Random r = new Random();
            int n = r.Next(2, 30);
            int m = r.Next(2, 30);
            int p = r.Next(2, 30);
            Matrix a = new float[n, m];
            Matrix b = new float[m, p];

            Matrix res = new float[n, p];
            for (int i = 0; i < a.D1; i++)
                for (int j = 0; j < a.D2; j++)
                    a[i, j] = r.Next(-10,10);
            for (int i = 0; i < b.D1; i++)
                for (int j = 0; j < b.D2; j++)
                    b[i, j] = r.Next(-10,10);

            for (int i = 0; i < a.D1; i++)
                for (int j = 0; j < a.D2; j++)
                    for (int k = 0; k < b.D2; k++)
                    {
                        res[i, k] += a[i, j] * b[j, k];
                    }

            var c = Matrix.MatrixMultiply(a, b);

            Assert.IsTrue(ArrayEqual<float>(res.Array, c.Array));
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
