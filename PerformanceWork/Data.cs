using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceWork
{
    public static class DataType
    {
        /// <summary>
        /// Data type of an array.
        /// </summary>
        public enum Type
        {
            Double,
            Float,
            Int32
        }

        /// <summary>
        /// Returns the byte size of the data type given.
        /// </summary>
        /// <param name="t">Data Type</param>
        /// <returns>Byte Size of the data type</returns>
        public static int GetByteSize(Type t)
        {
            if (t == Type.Double) return 8;
            else if (t == Type.Float | t == Type.Int32) return 4;
            throw new Exception("Undefined data type");
        }
    }
}
