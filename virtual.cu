#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

template<typename Base>
class device_copyable
{
public:
	virtual void copyToDevice(Base** pos) = 0;
	virtual ~device_copyable() = default;
};

template<typename Base, typename Derived>
__global__ void copy_to_device(Base** object, Derived h)
{
	*object = new Derived(h);
}

template<typename Base, typename Derived>
class implements_device_copyable : virtual public device_copyable<Base>
{
public:
	void copyToDevice(Base** pos) override
	{
		copy_to_device<Base, Derived><<<1,1>>>(pos, dynamic_cast<Derived&>(*this));
	}
};

class Base : public virtual device_copyable<Base>
{
public:
	int b_id;
public:
	__host__ __device__ Base(int _b_id):b_id(_b_id){}
	__host__ __device__ Base(const Base& b):b_id(b.b_id){}
	virtual ~Base() = default;
	__device__ virtual void f(){};
};

class Sub1 : public Base, public implements_device_copyable<Base, Sub1>
{
public:
	int sub_id;
public:
	__host__ __device__ Sub1(int _b_id, int _sub_id):Base(_b_id), sub_id(_sub_id){}
	__host__ __device__ Sub1(const Sub1& s):Base(s), sub_id(s.sub_id){}
	__device__ void f() override
	{
		printf("hello 1: %d %d!\n", b_id, sub_id);
	}
};

__global__ void runner(Base** objects, int len)
{
	for (int i=0; i<len; i++)
		objects[i]->f();
}

void run(std::vector<std::unique_ptr<Base>> objs)
{
	Base** d_objects;
	CUDA_CALL(cudaMalloc((void **)&d_objects, objs.size() * sizeof(Base*)));
	for (int i=0; i<objs.size(); i++)
	{
		objs[i]->copyToDevice(&d_objects[i]);
	}
	runner<<<1,1>>>(d_objects, objs.size());
}

int main()
{
	std::vector<std::unique_ptr<Base>> objs;
	objs.push_back(std::unique_ptr<Base>(new Sub1(42, 68)));
	run(std::move(objs));
	cudaDeviceSynchronize();
	return 0;
}
