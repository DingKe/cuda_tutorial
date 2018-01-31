#include <stdio.h> 
#include <stdlib.h>

__global__ void childKernel()
{ 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("Hello "); 
    }
} 

__global__ void parentKernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0)
    {
        return;
    }

    // launch child 	
    childKernel<<<10, 32>>>(); 

    if (cudaSuccess != cudaGetLastError()) 
    { 		
        printf("Child kernel failed to lauch\n");
        return; 
    } 

    // wait for child to complete 
    if (cudaSuccess != cudaDeviceSynchronize()) 
    {	
        printf("Child kernel failed to lauch\n");
        return; 
    } 
    printf("World!\n"); 
} 

int main(int argc, char *argv[]) 
{ 
    // launch parent 
    parentKernel<<<10, 32>>>(); 

    if (cudaSuccess != cudaGetLastError())
    { 		
        return 1; 
    } 
	
    // wait for parent to complete 
    if (cudaSuccess != cudaDeviceSynchronize())
    {		
        return 2; 
    } 	

    return 0;
}
