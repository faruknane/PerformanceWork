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
            Assert.IsTrue(ArrayEqual(v1, v2));
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
            Assert.IsTrue(ArrayEqual(v1, v2));
        }

        [TestMethod]
        public void Equality()
        {
            {
                float[] v1 = { 1, 2, 3 };
                float[] v2 = { 1, 2, 3 };
                bool res = Vectorization.ElementWiseIsEqualsAVX(v1, v2, v1.Length);
                bool res2 = true;
                Assert.AreEqual(res, res2);
            }
            {
                float[] v1 = { 1, 2, 2 };
                float[] v2 = { 1, 2, 3 };
                bool res = Vectorization.ElementWiseIsEqualsAVX(v1, v2, v1.Length);
                bool res2 = false;
                Assert.AreEqual(res, res2);
            }
        }
        [TestMethod]
        public unsafe void Add()
        {
            int l = 1000000;
            float[] v1 = new float[l];
            for (int i = 0; i < l; i++)
                v1[i] = i;
            float[] v2 = new float[l];
            for (int i = 0; i < l; i++)
                v2[i] = i;
            float[] res = new float[3];
            fixed (float* a = v1, b = v2, y = res)
                Vectorization.ElementWiseAddAVX(a, b, y, res.Length);
            float[] res2 = new float[l];
            for (int i = 0; i < l; i++)
                res2[i] = i * 2;
            Assert.IsTrue(ArrayEqual(res, res2));
        }
        [TestMethod]
        public unsafe void Add2()
        {
            float[] v1 = { 1, 2, 3 };
            float v2 = 2;
            float[] res = new float[3];
            fixed (float* a = v1, y = res)
                Vectorization.ElementWiseAddAVX(a, v2, y, res.Length);
            float[] res2 = { 3, 4, 5 };

            Assert.IsTrue(ArrayEqual(res, res2));
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
        public unsafe void Multiply()
        {
            int l = 1000000;
            float[] v1 = new float[l];
            for (int i = 0; i < l; i++)
                v1[i] = i;
            float[] v2 = new float[l];
            for (int i = 0; i < l; i++)
                v2[i] = i;
            float[] res = new float[3];
            fixed (float* a = v1, b = v2, y = res)
                Vectorization.ElementWiseMultiplyAVX(a, b, y, res.Length);
            float[] res2 = new float[l];
            for (int i = 0; i < l; i++)
                res2[i] = i*i;
            Assert.IsTrue(ArrayEqual(res, res2));
        }
        [TestMethod]
        public unsafe void Multiply2()
        {
            int l = 1000000;
            float[] v1 = new float[l];
            for (int i = 0; i < l; i++)
                v1[i] = i;

            float v2 = 3;
            float[] res = new float[3];
            fixed (float* a = v1, y = res)
                Vectorization.ElementWiseMultiplyAVX(a, v2, y, res.Length);
            float[] res2 = new float[l];
            for (int i = 0; i < l; i++)
                res2[i] = i*3;
            Assert.IsTrue(ArrayEqual(res, res2));
        }

        [TestMethod]
        public unsafe void MatrixMultiply()
        {

            for (int kk = 0; kk < 1000; kk++)
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
                        a[i, j] = r.Next(-10, 10);
                for (int i = 0; i < b.D1; i++)
                    for (int j = 0; j < b.D2; j++)
                        b[i, j] = r.Next(-10, 10);

                for (int i = 0; i < a.D1; i++)
                    for (int j = 0; j < a.D2; j++)
                        for (int k = 0; k < b.D2; k++)
                        {
                            res[i, k] += a[i, j] * b[j, k];
                        }

                var c = Matrix.MatrixMultiply(a, b);

                if (!Vectorization.ElementWiseIsEqualsAVX(res.Array, c.Array, res.D1 * res.D2))
                {
                    throw new Exception("Eþit Deðil!");
                }
                a.Dispose();
                b.Dispose();
                c.Dispose();
                res.Dispose();
            }
        }
        [TestMethod]
        public unsafe void MatrixMultiply2()
        {
            for (int kk = 0; kk < 1000; kk++)
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
                        a[i, j] = r.Next(-10, 10);
                for (int i = 0; i < b.D1; i++)
                    for (int j = 0; j < b.D2; j++)
                        b[i, j] = r.Next(-10, 10);

                for (int i = 0; i < a.D1; i++)
                    for (int j = 0; j < a.D2; j++)
                        for (int k = 0; k < b.D2; k++)
                        {
                            res[i, k] += a[i, j] * b[j, k];
                        }

                //var c = Matrix.MatrixMultiply(a, b);

                Matrix c = new Matrix(a.D1, b.D2);
                Vectorization.MatrixMultiply(a.GetPointer(), a.D1, a.D2, b.GetPointer(), b.D1, b.D2, c.GetPointer());

                if (!Vectorization.ElementWiseIsEqualsAVX(res.Array, c.Array, res.D1 * res.D2))
                {
                    throw new Exception("Eþit Deðil!");
                }
                a.Dispose();
                b.Dispose();
                c.Dispose();
                res.Dispose();
            }
        }

        [TestMethod]
        public unsafe void SigmoidTest()
        {
            float[] v1 = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            float[] res = new float[9];
            fixed (float* ptr_v1 = v1, ptr_res = res)
                Vectorization.Sigmoid(ptr_v1, ptr_res, v1.Length);
            float[] res2 = { 0.731058359f, 0.8807941f, 0.95257f, 0.731058359f, 0.8807941f, 0.95257f, 0.731058359f, 0.8807941f, 0.95257f };
            Assert.IsTrue(ArrayEqual(res, res2));
        }

        [TestMethod]
        public unsafe void SumOfVectorTest()
        {
            float[] v1 = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            float res = 0;
            fixed (float* ptr_v1 = v1)
                res = Vectorization.SumOfVector(ptr_v1, v1.Length);
            float res2 = 18;
            Assert.IsTrue(res == res2);
        }

        [TestMethod]
        public unsafe void ElementWise_A_MultipliedBy_1_Minus_ATest()
        {
            float[] v1 = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            fixed (float* ptr_v1 = v1)
                Vectorization.ElementWise_A_MultipliedBy_1_Minus_A(ptr_v1, ptr_v1, v1.Length);
            float[] res2 = { 0, -2, -6 , 0, -2, -6 , 0, -2, -6 };
            Assert.IsTrue(ArrayEqual(v1, res2));
        }

        [TestMethod]
        public unsafe void ElementWiseMultiplyAndReturnSumTest()
        {
            float[] v1 = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            float[] v2 = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            float res = 0;
            fixed (float* ptr_v1 = v1, ptr_v2 = v2)
                res = Vectorization.ElementWiseMultiplyAndReturnSum(ptr_v1, ptr_v2, ptr_v1, v1.Length);
            float res2 = 42;
            Assert.IsTrue(res == res2);
            float[] res3 = { 1, 4, 9 , 1, 4, 9 , 1, 4, 9};
            Assert.IsTrue(ArrayEqual(v1, res3));
        }

        [TestMethod]
        public unsafe void SumOfPerColumnTest()
        {
            Matrix a = new Matrix(626, 100);
            Matrix c = new Matrix(1, 100);
            for (int i = 0; i < a.D1; i++)
                for (int j = 0; j < a.D2; j++)
                    a[i, j] = i;
            //sum = 625*626/2
            Vectorization.SumOfPerColumn(a.Array, c.Array, a.D1, a.D2);
            float res2 = 625 * 626 / 2;

            bool d = true;
            for (int j = 0; j < a.D2; j++)
                d = d & (c[0, j] == res2);
            Assert.IsTrue(d);
        }
        public bool ArrayEqual(float[] v1, float[] v2)
        {
            return Vectorization.ElementWiseIsEqualsAVX(v1,v2,v1.Length);
        }
    }
}
