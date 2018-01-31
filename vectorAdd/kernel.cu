#include "cuda.h"

__global__ void addOneKernel(float* out, const float* in, int numElements)
{  	
    int stride = blockDim.x * gridDim.x;
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;	
    for (; tidx < numElements; tidx += stride)
    {
        out[tidx] = in[tidx] + 1;   		
    }		
}

// Kernel Wrapper
void addOne(float* out_h, const float* in_h, int numElements)
{ 
    float *in_d, *out_d;	
    size_t size = sizeof(float) * numElements;    
    cudaMalloc(&in_d, size);
    cudaMalloc(&out_d, size);

    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    addOneKernel<<<blocksPerGrid, threadsPerBlock>>>(out_d, in_d, numElements);	

    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);	
}
