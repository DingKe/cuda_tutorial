#include "utils.h"

__inline__ __device__
int warpReduceSumDeviceKernel(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset);
    return val;
}

__global__ void  warpReduceSumKernel(float *out, const float *in, int numElementsPerBlock, int numElements)
{
    // Shared memory for 32 partial sums
    extern __shared__ int shared[];

    int offset = numElementsPerBlock * blockIdx.x;
    in += offset;
    int numElementsReduced = offset + numElementsPerBlock < numElements ? numElementsPerBlock : numElements - offset; 

    int tidx = threadIdx.x;

    int lane = tidx & 0x1f;
    int wid = tidx >> 5;

    float val = globalReduceSumDeviceKernel2(in, numElementsReduced); 
    __syncthreads();

    // each warp performs partial reduction
    val = warpReduceSumDeviceKernel(val);

    // write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // read from shared memory only if that warp existed
    val = (lane < blockDim.x >> 5) ? shared[lane] : 0;

    // final reduce within first warp
    if (wid == 0)
    {	
        val = warpReduceSumDeviceKernel(val);
        if (tidx == 0) *out = val;
    }
}

void warpReduceSum(float *out, const float *in, int numElements)
{
    /*
    int numElementsPerBlock = min(numElements, REDUCE_SIZE);
    int numThreadsPerBlock = BLOCK_SIZE;
    int numWarpsPerBlock = numThreadsPerBlock / WARP_SIZE;
    int numBlocksPerGrid = (numElements + numElementsPerBlock - 1) / numElementsPerBlock;
    */
    int numThreadsPerBlock = BLOCK_SIZE;
    int numWarpsPerBlock = numThreadsPerBlock / WARP_SIZE;
    int numBlocksPerGrid = GRID_SIZE;
    int numElementsPerBlock = (numElements + numBlocksPerGrid - 1) / numBlocksPerGrid;
    numElementsPerBlock = numElementsPerBlock / 4 * 4; // divided by 4
    numElementsPerBlock = max(numElementsPerBlock, REDUCE_SIZE);
    numBlocksPerGrid = (numElements + numElementsPerBlock - 1) / numElementsPerBlock;

    float * out_buffer;
    cudaMalloc(&out_buffer, sizeof(float) * numBlocksPerGrid);
    warpReduceSumKernel<<<numBlocksPerGrid, numThreadsPerBlock, numWarpsPerBlock>>>(out_buffer, in, numElementsPerBlock, numElements);
    cudaDeviceSynchronize();
    warpReduceSumKernel<<<1, WARP_SIZE, 1>>>(out, out_buffer, numElementsPerBlock, numBlocksPerGrid);
    cudaFree(out_buffer);
}
