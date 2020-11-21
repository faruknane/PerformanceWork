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
        public static ArrayPool Host = new ArrayPool(30000000, DeviceConfig.Host_Unkown);

        public static List<ArrayPool> Gpu = new List<ArrayPool>();

        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool GetDevicePool(DeviceConfig device)
        {
            if (device.DevType == DeviceConfig.DeviceType.NvidiaGPU)
                return GetNvidiaGpuPool(device.DeviceID);
            else if (device.DevType == DeviceConfig.DeviceType.Host)
                return Host;
            else
                throw new Exception("Unsupported Device!");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ArrayPool GetNvidiaGpuPool(int deviceId)
        {
            for (int i = Gpu.Count; i <= deviceId; i++)
                Gpu.Add(new ArrayPool(30000000, DeviceConfig.NvidiaGPU_Unknown));
            return Gpu[deviceId];
        }
    }
}
