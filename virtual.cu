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
public:
	Base() = default;
	__host__ __device__ Base(int _b_id):b_id(_b_id){}
	virtual ~Base() = default;
};

class BaseDevice;

class BaseHost : public virtual Base
{
public:
	virtual void createOnDevice(BaseDevice** pos) = 0;
};

class BaseDevice : public virtual Base
{
public:
	__device__ virtual void f() = 0;
};

class Sub1 : public virtual Base
{
public:
	int sub_id;
public:
	__host__ __device__ Sub1(int _sub_id):sub_id(_sub_id){}
};

class Sub1Host : public BaseHost, public Sub1
{
public:
	Sub1Host(int _b_id, int _sub_id):Base(_b_id), Sub1(_sub_id){}
	void createOnDevice(BaseDevice** pos) override;
};

class Sub1Device : public BaseDevice, public Sub1
{
public:
	__device__ Sub1Device(int _b_id, int _sub_id):Base(_b_id), Sub1(_sub_id){}
public:
	__device__ void f() override
	{
		printf("hello 1: %d %d!\n", b_id, sub_id);
	}
};

__global__ void create_sub1(BaseDevice** object, int base_id, int sub_id)
{
	*object = new Sub1Device(base_id, sub_id);
}

void Sub1Host::createOnDevice(BaseDevice** pos)
{
	create_sub1<<<1,1>>>(pos, b_id, sub_id);
}

__global__ void runner(BaseDevice** objects, int len)
{
	for (int i=0; i<len; i++)
		objects[i]->f();
}

void run(std::vector<std::unique_ptr<BaseHost>> objs)
{
	BaseDevice** d_objects;
	CUDA_CALL(cudaMalloc((void **)&d_objects, objs.size() * sizeof(BaseDevice*)));
	for (int i=0; i<objs.size(); i++)
	{
		objs[i]->createOnDevice(&d_objects[i]);
	}
	runner<<<1,1>>>(d_objects, objs.size());
}

int main()
{
	std::vector<std::unique_ptr<BaseHost>> funcs;
	funcs.push_back(std::unique_ptr<BaseHost>(new Sub1Host(42, 68)));
	run(std::move(funcs));
	cudaDeviceSynchronize();
	return 0;
}
