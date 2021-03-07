using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics.Pool
{
    public static class TensorPool
    {
        public const int PoolSize = 10000000*4+1;

        public static ArrayPool Host = new ArrayPool(PoolSize, Device.Host);

        public static List<ArrayPool> Gpu = new List<ArrayPool>();

        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool GetDevicePool(Device device)
        {
            if (device.Type == DeviceType.NvidiaGPU)
                return GetNvidiaGpuPool(device.ID);
            else if (device.Type == DeviceType.Host)
                return Host;
            else
                throw new Exception("Unsupported Device!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool GetNvidiaGpuPool(int deviceId)
        {
            for (int i = Gpu.Count; i <= deviceId; i++)
                Gpu.Add(new ArrayPool(PoolSize, Device.Nvidia(Gpu.Count)));
            return Gpu[deviceId];
        }
    }
}
