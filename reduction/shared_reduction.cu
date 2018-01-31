#include "utils.h"

__device__ void sharedReduceSumDeviceKernel(float *out, const float *in, float *buffer, int numElements)
{
    int tidx = threadIdx.x;

    buffer[tidx] = globalReduceSumDeviceKernel2(in, numElements);
    __syncthreads();

#pragma unroll
    for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1)
    {
        if (tidx < i)
        {
            buffer[tidx] += buffer[tidx + i];
        }
        __syncthreads();
    }

    if (tidx == 0) *out = *buffer;
}

__global__ void sharedReduceSumKernel(float *out, const float *in, int numElementsPerBlock, int numElements)
{
    static __shared__ float buffer[BLOCK_SIZE];

    int offset = numElementsPerBlock * blockIdx.x;
    int numElementsReduced = offset + numElementsPerBlock <= numElements ? numElementsPerBlock : numElements - offset; 

    sharedReduceSumDeviceKernel(out + blockIdx.x, in + offset, buffer, numElementsReduced);
}

void sharedReduceSum(float *out, const float *in, int numElements)
{
    int numThreadsPerBlock = BLOCK_SIZE;
    int numBlocksPerGrid = GRID_SIZE;
    int numElementsPerBlock = (numElements + numBlocksPerGrid - 1) / numBlocksPerGrid;
    numElementsPerBlock = numElementsPerBlock / 4 * 4; // divided by 4
    numElementsPerBlock = max(numElementsPerBlock, REDUCE_SIZE);
    numBlocksPerGrid = (numElements + numElementsPerBlock - 1) / numElementsPerBlock;

    float *out_buffer;
    cudaMalloc(&out_buffer, sizeof(float) * numBlocksPerGrid);

    sharedReduceSumKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(out_buffer, in, numElementsPerBlock, numElements);
    cudaDeviceSynchronize();
    sharedReduceSumKernel<<<1, numThreadsPerBlock>>>(out, out_buffer, numBlocksPerGrid, numBlocksPerGrid);

    cudaFree(out_buffer);
}
