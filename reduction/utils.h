#ifndef __UTILS_H__
#define __UTILS_H__

#define WARP_SIZE (32)
#define BLOCK_SIZE (1024)
#define GRID_SIZE (13)
#define REDUCE_SIZE (BLOCK_SIZE * 100)


__inline__ __device__ float globalReduceSumNaiveDeviceKernel(const float *in, int numElements)
{
    int tidx = threadIdx.x;

    float sum = 0;
    for (int i = tidx; i < numElements; i += blockDim.x)
    {
        sum += in[i];
    }
    return sum;
}

__inline__ __device__ float globalReduceSumDeviceKernel(const float *in, int numElements)
{
    int tidx = threadIdx.x;

    float sum = 0;
    for (int i = tidx; i < numElements / 4; i += blockDim.x)
    {
        float4 f4 = ((float4*)in)[i];
        sum += f4.x + f4.y + f4.z + f4.w;
    }
    /* corner case*/
    for (int i = numElements / 4 * 4 + tidx; i < numElements; i += blockDim.x)
    {
        sum += in[i];
    }
    return sum;
}

__inline__ __device__ float globalReduceSumDeviceKernel2(const float *in, int numElements)
{
    int tidx = threadIdx.x;

    float sum = 0;
    if (numElements % 4 == 0)
    {
        for (int i = tidx; i < numElements / 4; i += blockDim.x)
        {
            float4 f4 = ((float4*)in)[i];
            sum += f4.x + f4.y + f4.z + f4.w;
        }
    } else {
        for (int i = tidx; i < numElements; i += blockDim.x)
        {
            sum += in[i];
        }
    }
    return sum;
}

#endif
