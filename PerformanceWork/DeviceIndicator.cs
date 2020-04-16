using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork
{
    public struct DeviceIndicator
    {
        public DeviceType Type { get; set; }
        public int DeviceID { get; set; }

        public DeviceIndicator(DeviceType dev, int devid)
        {
            this.Type = dev;
            this.DeviceID = devid;
        }

        public static DeviceIndicator Gpu(int devid)
        {
            return new DeviceIndicator(DeviceType.Gpu, devid);
        }
        public static DeviceIndicator Host()
        {
            return new DeviceIndicator(DeviceType.Host, -1);
        }

        public static bool operator ==(DeviceIndicator b1, DeviceIndicator b2)
        {
            return b1.Type == b2.Type && (b1.Type == DeviceType.Host || b1.DeviceID == b2.DeviceID);
        }

        public static bool operator !=(DeviceIndicator b1, DeviceIndicator b2)
        {
            return b1.Type != b2.Type || (b1.Type != DeviceType.Host && b1.DeviceID != b2.DeviceID);
        }
    }
}
