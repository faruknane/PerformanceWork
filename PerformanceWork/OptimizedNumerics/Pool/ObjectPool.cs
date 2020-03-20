using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace PerformanceWork.OptimizedNumerics.Pool
{

    //thread safe bir class mı?
    public unsafe class ObjectPool<T> : IDisposable
    {
        public int Count { get => Stack.Count; }
        public int UnreturnedCount { get; private set; } = 0;
        public Stack<T> Stack;

        private object Mutex = new object();

        public ObjectPool()
        {
            Stack = new Stack<T>();
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public T Rent()
        {
            lock (Mutex)
            {
                UnreturnedCount++;
                if (Stack.Count > 0)
                    return Stack.Pop();
                return default;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Return(T x)
        {
            lock (Mutex)
            {
                Stack.Push(x);
                UnreturnedCount--;
            }
        }

        public void Dispose()
        {
            lock (Mutex)
            {
                Stack.Clear();
            }
        }
    }
}
