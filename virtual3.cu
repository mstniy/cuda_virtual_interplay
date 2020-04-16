#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

class BaseHost
{
public:
	virtual ~BaseHost() = default;
};

class BaseDevice
{
public:
	__device__ BaseDevice() = default;
	__device__ virtual ~BaseDevice() = default;
	__device__ virtual void f() = 0;
};

class Sub1Host : public BaseHost
{
public:
	int x;
public:
	Sub1Host(int _x):x(_x)
	{
	}
	virtual ~Sub1Host() = default;
};

class Sub1Device : public BaseDevice
{
public:
	int x;
	__device__ Sub1Device(int _x):x(_x)
	{
	}
	__device__ virtual ~Sub1Device() = default;
public:
	__device__ void f() override
	{
		printf("hello 1: %d!\n", x);
	}
};

class Sub2Host : public BaseHost
{
public:
	char ch;
public:
	Sub2Host(char _ch):ch(_ch)
	{
	}
	virtual ~Sub2Host() = default;
};

class Sub2Device : public BaseDevice
{
public:
	char ch;
	__device__ Sub2Device(char _ch):ch(_ch)
	{
	}
	__device__ virtual ~Sub2Device() = default;
public:
	__device__ void f() override
	{
		printf("hello 2: %c!\n", ch);
	}
};

__global__ void runner(BaseDevice** objects, int len)
{
	for (int i=0; i<len; i++)
		objects[i]->f();
}

__global__ void create_sub1(BaseDevice** object, int x)
{
	*object = new Sub1Device(x);
}

__global__ void create_sub2(BaseDevice** object, char ch)
{
	*object = new Sub2Device(ch);
}

void run(std::vector<std::unique_ptr<BaseHost>> funcs)
{
	BaseDevice** d_objects;
	int index=0;
	CUDA_CALL(cudaMalloc((void **)&d_objects, funcs.size() * sizeof(BaseDevice*)));
	for (const auto& f : funcs)
	{
		if (dynamic_cast<Sub1Host*>(f.get()) != nullptr)
			create_sub1<<<1,1>>>(&d_objects[index], dynamic_cast<Sub1Host*>(f.get())->x);
		else if (dynamic_cast<Sub2Host*>(f.get()) != nullptr)
			create_sub2<<<1,1>>>(&d_objects[index], dynamic_cast<Sub2Host*>(f.get())->ch);
		else
		{
			std::cout << "Warning: Ignoring unknown object." << std::endl;
			index--;
		}
		index++;
	}
	runner<<<1,1>>>(d_objects, index);
}

int main()
{
	std::vector<std::unique_ptr<BaseHost>> funcs;
	funcs.push_back(std::unique_ptr<BaseHost>(new Sub1Host(42)));
	funcs.push_back(std::unique_ptr<BaseHost>(new Sub2Host('h')));
	run(std::move(funcs));
	cudaDeviceSynchronize();
	return 0;
}
