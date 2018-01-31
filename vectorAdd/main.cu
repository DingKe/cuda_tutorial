#include <stdio.h>
#include <stdlib.h>

// forward declearation
void addOne(float *out_h, const float *in_h, int numElements);

int main(void)
{   	
    int numElements = 50000;

    float *in_h, *out_h;
    in_h = (float *)malloc(sizeof(float) * numElements);
    out_h = (float *)malloc(sizeof(float) * numElements);

    for (int i = 0; i < numElements; ++i)
    {
        in_h[i] = rand() / (float)RAND_MAX;
    }	

    addOne(out_h, in_h, numElements);		

    float delta = 0;
    for(int i = 0; i < numElements; ++i)
    {
        delta += out_h[i] - in_h[i];
    }	
    printf("num: %d, delta: %.1f\n", numElements, delta);

    free(in_h);
    free(out_h);

    return 0;	
}
