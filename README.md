Under development!

# PerformanceWork Library
This library is dependent on Intel MKL, Nvidia Cuda 11.1, Cutensor 1.2.2.5. Works on both CPU and Nvidia GPUs. Currently, native codes are compiled for windows 10 only. 

### Hardware Requirements
#### For GPUs: 
- Nvidia compute_52, sm_52; compute_60, sm_60; compute_61, sm_61; compute_70, sm_70; compute_75, sm_75; compute_80, sm_80; compute_86, sm_86;
#### For CPUs: 
- x86-64 assembly support (for intel mkl and native c++ codes)
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
