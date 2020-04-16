#include <stdio.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

class Base
{
public:
	__device__ virtual void f() = 0;
};

class Sub : public Base
{
public:
	__device__ void f() override
	{
		printf("hello!\n");
	}
};

template<typename T>
__global__ void constructor(T* obj)
{
	new (obj) T;
}

__global__ void runner(Base* obj)
{
	obj->f();
}

int main()
{
	Sub* d_sub;
	CUDA_CALL(cudaMalloc((void **)&d_sub, 1 * sizeof(Sub)));
	constructor<Sub><<<1,1>>>(d_sub);
	runner<<<1,1>>>(d_sub);
	cudaDeviceSynchronize();
	return 0;
}
