using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork
{
    /// <summary>
    /// Data type.
    /// </summary>
    public enum NumberType
    {
        Float64,
        Float32,
        Float16,
        Int32,
        Int16,
        Uknown
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

    public struct TensorConfig
    {
        public static TensorConfig Host_Unkown = new TensorConfig(Device.Host, NumberType.Uknown);
        public static TensorConfig NvidiaGPU_Unkown = new TensorConfig(Device.Nvidia(0), NumberType.Uknown);

        public static TensorConfig NvidiaGPU_Float32 = new TensorConfig(Device.Nvidia(0), NumberType.Float32);

        public static TensorConfig Host_Float64 = new TensorConfig(Device.Host, NumberType.Float64);
        public static TensorConfig Host_Float32 = new TensorConfig(Device.Host, NumberType.Float32);
        public static TensorConfig Host_Int32 = new TensorConfig(Device.Host, NumberType.Int32);

        public NumberType NumType;
        public Device Device;

        public TensorConfig(Device device, NumberType datatype)
        {
            Device = device;
            NumType = datatype;
        }

        public TensorConfig(DeviceType devicetype, int devid, NumberType datatype)
        {
            Device = new Device();
            Device.Type = devicetype;
            Device.ID = devid;
            NumType = datatype;
        }

        public TensorConfig(DeviceType devicetype, NumberType datatype)
        {
            Device = new Device();
            Device.Type = devicetype;
            Device.ID = 0;
            NumType = datatype;
        }

        public int GetUnitLength()
        {
            return TensorConfig.GetUnitLength(this.NumType);
        }

        public static bool operator ==(TensorConfig b1, TensorConfig b2)
        {
            return b1.NumType == b2.NumType && b1.Device == b2.Device;
        }

        public static bool operator !=(TensorConfig b1, TensorConfig b2)
        {
            return !(b1 == b2);
        }

        public override bool Equals(object obj)
        {
            if (obj is TensorConfig d)
                return this == d;
            else
                return false;
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        /// Returns the byte size of the data type given.
        /// </summary>
        /// <param name="t">Data Type</param>
        /// <returns>Byte Size of the data type</returns>
        public static int GetUnitLength(NumberType t)
        {
            if (t == NumberType.Float64) return 8;
            else if (t == NumberType.Float32 | t == NumberType.Int32) return 4;
            else if (t == NumberType.Int16 || t == NumberType.Float16) return 2;
            throw new Exception("Undefined Number Type!");
        }
    }
}
