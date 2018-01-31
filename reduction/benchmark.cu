#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>

#define NUM_ELEMENTS (1024 * 1024 * 10)
#define RUN (100)

void globalReduceSum(float *out, const float *in, int numElements);
void sharedReduceSum(float *out, const float *in, int numElements);
void warpReduceSum(float *out, const float *in, int numElements);

void runTest(const char *label, void(*fptr) (float *, const float *, int), float *out, const float *in, int numElements)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up
    for (int i = 0; i < 10; i++)
    {
        fptr(out, in, numElements);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < RUN; i++)
    {
        fptr(out, in, numElements);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    float time_s = time_ms / (float)1e3;

    float GB = (float) numElements * sizeof(float) * RUN;
    float GBs = GB / time_s / (float)1e9;


    printf("%s. Time: %f, GB/s: %f\n", label, time_s, GBs);
}

int main(int argc, char **argv)
{
    float *out_d, *in_d, *out_h, *in_h;
    int numElements = NUM_ELEMENTS;

    in_h = (float *)malloc(sizeof(float) * numElements);
    out_h = (float *)malloc(sizeof(float) * numElements);
    for (int i = 0; i < numElements; ++i)
    {
        in_h[i] = 1.f;
    }	

    cudaMalloc(&out_d, sizeof(float));
    cudaMalloc(&in_d, sizeof(float) * numElements);

    cudaMemcpy(in_d, in_h, sizeof(float) * numElements, cudaMemcpyHostToDevice);

    cudaMemset(out_d, 0, sizeof(float));
    runTest("Reduce with global memory", globalReduceSum, out_d, in_d, numElements);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum: %.4f\n", *out_h);

    cudaMemset(out_d, 0, sizeof(float));
    runTest("Reduce with shared memory", sharedReduceSum, out_d, in_d, numElements);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum: %.4f\n", *out_h);

    cudaMemset(out_d, 0, sizeof(float));
    runTest("Reduce with warp shuffle", warpReduceSum, out_d, in_d, numElements);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum: %.4f\n", *out_h);

    double sum = 0;
    for (int i = 0; i < numElements; ++i)
    {
        sum += in_h[i];
    }
    printf("num: %d, reference sum: %f\n", numElements, sum);

    cudaFree(out_d);
    cudaFree(in_d);
    free(out_h);
    free(in_h);

    return 0;
}


