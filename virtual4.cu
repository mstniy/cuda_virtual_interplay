#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

class Base
{
public:
	int b_id;
	__host__ __device__ Base(int _b_id):b_id(_b_id){}
};

//TODO: We don't really need this
class BaseHost : public Base
{
public:
	using Base::Base;
	virtual ~BaseHost() = default;
};

class BaseDevice : public Base
{
public:
	using Base::Base;
	__device__ virtual ~BaseDevice() = default;
	__device__ virtual void f() = 0;
};

class Sub1
{
public:
	int sub_id;
	__host__ __device__ Sub1(int _sub_id):sub_id(_sub_id){}
};

class Sub1Host : public BaseHost, public Sub1
{
public:
	Sub1Host(int _b_id, int _sub_id):BaseHost(_b_id), Sub1(_sub_id){}
	virtual ~Sub1Host() = default;
};

class Sub1Device : public BaseDevice, public Sub1
{
public:
	__device__ Sub1Device(int _b_id, int _sub_id):BaseDevice(_b_id), Sub1(_sub_id){}
	__device__ virtual ~Sub1Device() = default;
public:
	__device__ void f() override
	{
		printf("hello 1: %d %d!\n", b_id, sub_id);
	}
};

__global__ void runner(BaseDevice** objects, int len)
{
	for (int i=0; i<len; i++)
		objects[i]->f();
}

__global__ void create_sub1(BaseDevice** object, int base_id, int sub_id)
{
	*object = new Sub1Device(base_id, sub_id);
}

void run(std::vector<std::unique_ptr<BaseHost>> funcs)
{
	BaseDevice** d_objects;
	int index=0;
	CUDA_CALL(cudaMalloc((void **)&d_objects, funcs.size() * sizeof(BaseDevice*)));
	for (const auto& f : funcs)
	{
		if (dynamic_cast<Sub1Host*>(f.get()) != nullptr)
			create_sub1<<<1,1>>>(&d_objects[index], dynamic_cast<Sub1Host*>(f.get())->b_id, dynamic_cast<Sub1Host*>(f.get())->sub_id);
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
	funcs.push_back(std::unique_ptr<BaseHost>(new Sub1Host(42, 68)));
	run(std::move(funcs));
	cudaDeviceSynchronize();
	return 0;
}
