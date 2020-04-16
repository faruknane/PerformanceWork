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
        public static ArrayPool Host = ArrayPool.Create(30000000, 300000);
        public static List<ArrayPool> Gpu = new List<ArrayPool>();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool GetDevicePool(DeviceIndicator device)
        {
            if (device.Type == DeviceType.Gpu)
                return GetGpuPool(device.DeviceID);
            else if (device.Type == DeviceType.Host)
                return Host;
            else
                throw new Exception("Unsupported Device!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool GetGpuPool(int deviceId)
        {
            for (int i = Gpu.Count; i <= deviceId; i++)
                Gpu.Add(ArrayPool.Create(30000000, 300000, true, i));
            return Gpu[deviceId];
        }
    }
}
