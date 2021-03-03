using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork
{
    public struct DeviceConfig
    {
        /// <summary>
        /// Data type.
        /// </summary>
        public enum NumberType
        {
            CPUDouble,
            CPUFloat32,
            CPUInt32,
            Uknown
        }

        public enum DeviceType
        {
            Host,
            NvidiaGPU
        }
        public static DeviceConfig Host_Unkown = new DeviceConfig(DeviceType.Host, 0, NumberType.Uknown);
        public static DeviceConfig NvidiaGPU_Unkown = new DeviceConfig(DeviceType.NvidiaGPU, 0, NumberType.Uknown);
        
        public static DeviceConfig Host_Float32 = new DeviceConfig(DeviceType.Host, 0, NumberType.CPUFloat32);
        public static DeviceConfig Host_Double = new DeviceConfig(DeviceType.Host, 0, NumberType.CPUDouble);
        public static DeviceConfig Host_Int32 = new DeviceConfig(DeviceType.Host, 0, NumberType.CPUInt32);

        public NumberType NumType { get; set; }
        public DeviceType DevType { get; set; }
        public int DeviceID { get; set; }
        public static DeviceConfig NvidiaGPU_Unknown { get; internal set; }

        public DeviceConfig(DeviceType devicetype, int devid, NumberType datatype)
        {
            NumType = datatype;
            DeviceID = devid;
            DevType = devicetype;
        }

        public int GetUnitLength()
        {
            return DeviceConfig.GetUnitLength(this.NumType);
        }

        public static bool operator ==(DeviceConfig b1, DeviceConfig b2)
        {
            return b1.NumType == b2.NumType && b1.DevType == b2.DevType && b1.DeviceID == b2.DeviceID;
        }

        public static bool operator !=(DeviceConfig b1, DeviceConfig b2)
        {
            return !(b1.NumType == b2.NumType && b1.DevType == b2.DevType && b1.DeviceID == b2.DeviceID);
        }

        public override bool Equals(object obj)
        {
            if (obj is DeviceConfig d)
            {
                return this == d;
            }
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
            if (t == NumberType.CPUDouble) return 8;
            else if (t == NumberType.CPUFloat32 | t == NumberType.CPUInt32) return 4;
            throw new Exception("Undefined Number Type!");
        }
    }
}
