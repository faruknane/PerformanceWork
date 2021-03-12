Under development!

# PerformanceWork Library
This library is dependent on Intel MKL, Nvidia Cuda 11.1, Cutensor 1.2.2.5. Works on both CPU and Nvidia GPUs. Currently, native codes are compiled for windows 10 only. 

### Hardware Requirements
#### For GPUs: 
- Nvidia > sm50
#### For CPUs: 
- Avx2 support

## Supported Tensor Operations
- Element-Wise Add / Subtract / Multiply / Divide
- Element-Wise Power (CPU-Only)
- Relu (CPU-Only)
- Sigmoid (CPU-Only)
- Softmax (CPU-Only)
- Matrix Multiplication
- Einsum Operation (GPU-Only)
- Expand and Shrink Tensor (CPU-Only)

## Supported Gradient Tensor Operations
- Element-Wise Add / Subtract / Multiply / Divide
- Element-Wise Power (CPU-Only)
- Relu (CPU-Only)
- Sigmoid (CPU-Only)
- Softmax (CPU-Only)
- Matrix Multiplication
- Expand and Shrink Tensor (CPU-Only)
