#include "utils.h"

__global__ void globalReduceSumKernel(float *out, const float *in, int numElementsPerBlock, int numElements)
{
    int offset = numElementsPerBlock * blockIdx.x;
    numElementsPerBlock = offset + numElementsPerBlock <= numElements ? numElementsPerBlock : numElements - offset; 

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    out[tidx] = globalReduceSumDeviceKernel2(in + offset, numElementsPerBlock);
}

void globalReduceSum(float *out, const float *in, int numElements)
{
    int numThreadsPerBlock = BLOCK_SIZE;
    int numBlocksPerGrid = GRID_SIZE;
    int numElementsPerBlock = (numElements + numBlocksPerGrid - 1) / numBlocksPerGrid;
    numElementsPerBlock = numElementsPerBlock / 4 * 4; // divided by 4
    numElementsPerBlock = max(numElementsPerBlock, REDUCE_SIZE);
    numBlocksPerGrid = (numElements + numElementsPerBlock - 1) / numElementsPerBlock;

    float *outBuffer;
    int bufferSize = numBlocksPerGrid * numThreadsPerBlock;
    cudaMalloc(&outBuffer, sizeof(float) * bufferSize);

    globalReduceSumKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(outBuffer, in, numElementsPerBlock, numElements);
    cudaDeviceSynchronize();
    if (bufferSize > 1024)
    {
        globalReduceSum(out, outBuffer, bufferSize);
    } else {
        globalReduceSumKernel<<<1, 1>>>(out, outBuffer, 1024, bufferSize);
    }
    cudaFree(outBuffer);
}
