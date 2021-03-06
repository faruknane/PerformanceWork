﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics.Tensors
{
    public static class TensorExtension
    {
        /// <summary>
        ///  Creates a tensor on Host Device. It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="s">Shape for the tensor to be created.</param>
        /// <param name="type">NumberType for the tensor to be created.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe Tensor ToDisposedTensor(this Array data, Shape s, NumberType type)
        {
            return Tensor.ToDisposedTensor(data, s, type);
        }

        /// <summary>
        ///  Creates a tensor on Host Device. It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="s">Shape for the tensor to be created.</param>
        /// <param name="type">NumberType for the tensor to be created.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe Tensor ToDisposedTensor(this Array data, NumberType type)
        {
            Shape s = Shape.DimensionOf(data.Rank);
            for (int i = 0; i < data.Rank; i++)
                s.Dimensions[i] = data.GetLength(i);
            s.CalculateMultiplied();

            return Tensor.ToDisposedTensor(data, s, type);
        }

        /// <summary>
        ///  Creates a tensor on Host Device. It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="s">Shape for the tensor to be created.</param>
        /// <param name="type">NumberType for the tensor to be created.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe Tensor ToDisposedTensor(this Array data, Shape s)
        {
            NumberType type = NumberTypeHelper.GetType(data.GetType().GetElementType());

            if (type != NumberType.Unkown)
                return Tensor.ToDisposedTensor(data, s, type);
            else
                throw new Exception("Unsupported Array Type!");
        }

        /// <summary>
        ///  Creates a tensor on Host Device. It assumes that the new tensor is already returned. So, It won't do anything if the tensor gets disposed.
        /// </summary>
        /// <param name="s">Shape for the tensor to be created.</param>
        /// <param name="type">NumberType for the tensor to be created.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe Tensor ToDisposedTensor(this Array data)
        {
            NumberType type = NumberTypeHelper.GetType(data.GetType().GetElementType());

            if (type != NumberType.Unkown)
                return data.ToDisposedTensor(type);
            else
                throw new Exception("Unsupported Array Type!");
        }

    }
}
