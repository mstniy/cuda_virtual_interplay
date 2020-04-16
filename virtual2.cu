#include <stdio.h>
#include <vector>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

#define make_device_instantiable(_BASE, _CLASS) __device__ _BASE* (*_CLASS ## _deviceCreate )() = createOnDevice<_BASE, _CLASS>
#define get_class_device_instantiator(_DEST, _CLASS) do{CUDA_CALL(cudaMemcpyFromSymbol(&_DEST, _CLASS ## _deviceCreate, sizeof(_CLASS ## _deviceCreate))); }while(0)

class Base
{
public:
	__device__ virtual void f() = 0;
};

template<typename A, typename B>
__device__ A* createOnDevice()
{
	return new B;
}

class Sub1 : public Base
{
public:
	__device__ void f() override
	{
		printf("hello 1!\n");
	}
};
make_device_instantiable(Base, Sub1);

class Sub2 : public Base
{
public:
	__device__ void f() override
	{
		printf("hello 2!\n");
	}
};
make_device_instantiable(Base, Sub2);

__global__ void runner(Base* (* obj)())
{
	obj()->f();
}

void run(std::vector<Base*(*)()> funcs)
{
	for (auto f : funcs)
		runner<<<1,1>>>(f);
}

int main()
{
	std::vector<Base*(*)()> funcs(2);
	get_class_device_instantiator(funcs[0], Sub1);
	get_class_device_instantiator(funcs[1], Sub2);
	run(funcs);
	cudaDeviceSynchronize();
	return 0;
}
