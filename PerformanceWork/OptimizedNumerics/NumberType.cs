using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork.OptimizedNumerics
{
    public static class NumberTypeHelper
    {
        public static NumberType GetType(Type t)
        {
            if (t == typeof(float)) return NumberType.Float32;
            if (t == typeof(double)) return NumberType.Float64;
            
            if (t == typeof(short)) return NumberType.Int16;
            if (t == typeof(int)) return NumberType.Int32;
            if (t == typeof(long)) return NumberType.Int64;

            return NumberType.Unkown;
        }
    }
    /// <summary>
    /// Data type.
    /// </summary>
    public enum NumberType
    {
        Float64,
        Float32,
        Float16,
        Int64,
        Int32,
        Int16,
        Unkown
    }

    public enum DeviceType
    {
        Host,
        NvidiaGPU,
        AmdGPU
    }

    public struct Device
    {
        public static Device Host { get; } = new Device() { ID = 0, Type = DeviceType.Host };
        public static Device Nvidia(int devid) => new Device() { ID = devid, Type = DeviceType.NvidiaGPU };

        public int ID;
        public DeviceType Type;

        public static bool operator ==(Device b1, Device b2)
        {
            return b1.ID == b2.ID && b1.Type == b2.Type;
        }

        public static bool operator !=(Device b1, Device b2)
        {
            return !(b1 == b2);
        }

        public override bool Equals(object obj)
        {
            if (obj is Device d)
                return this == d;
            else
                return false;
        }
    }
}
