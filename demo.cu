#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include "virtual_interplay.h"

#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)
#endif

using namespace std;

class Base : public virtual interplay_movable<Base>
{
public:
	int b_id;
public:
	Base() = default;
	__host__ __device__ Base(int _b_id):b_id(_b_id){}
	__host__ __device__ Base(const Base& b):b_id(b.b_id){}
	__host__ __device__ Base(Base&& b):b_id(b.b_id){}
	virtual ~Base() = default;
	__device__ virtual void f() = 0;
	__host__ __device__ virtual void g() = 0;
};

// Classes with virtual bases are supported
class Sub1 : virtual public Base, public implements_interplay_movable<Base, Sub1, true>
{
public:
	int sub_id;
public:
	Sub1() = default;
	__host__ __device__ Sub1(int _b_id, int _sub_id):Base(_b_id), sub_id(_sub_id){}
	__host__ __device__ Sub1(const Sub1& s):Base(s), sub_id(s.sub_id){}
	__device__ __host__ Sub1(Sub1&& s):Base(std::move(s)), sub_id(s.sub_id){}
	__device__ void f() override
	{
		printf("Sub1::f %d %d!\n", b_id, sub_id);
		b_id++;
		sub_id++;
	}
	__host__ __device__ void g() override
	{
		printf("Sub1::g %d %d!\n", b_id, sub_id);
	}
};

class Sub2 : public Base, public implements_interplay_movable<Base, Sub2>
{
public:
	char sub_ch;
public:
	Sub2() = default;
	__host__ __device__ Sub2(int _b_id, char _sub_ch):Base(_b_id), sub_ch(_sub_ch){}
	Sub2(const Sub2&) = delete; // non-copy constructible classes are also supported
	__device__ __host__ Sub2(Sub2&& s):Base(std::move(s)), sub_ch(s.sub_ch){}
	__device__ void f() override
	{
		printf("Sub2::f %d %c!\n", b_id, sub_ch);
		b_id++;
		sub_ch++;
	}
	__host__ __device__ void g() override
	{
		printf("Sub2::g %d %c!\n", b_id, sub_ch);
	}
};

__global__ void runner(Base** objects, int len)
{
	for (int i=0; i<len; i++)
	{
		objects[i]->f();
	}
}

void run(const std::vector<std::unique_ptr<Base>>& objs)
{
	std::vector<Base*> raw_objs;
	for (const auto& obj : objs)
		raw_objs.push_back(obj.get());
	ClassMigrator<Base> migrator(raw_objs);
	// Migrate the objects to the device
	Base** d_objs = migrator.toDevice();
	// Run the demo to make sure everything works.
	runner<<<1,1>>>(d_objs, objs.size());
	// Migrate the objects back to the host
	migrator.toHost();
	// Another demo to make sure that the changes come back to the host
	for (Base* obj : raw_objs)
	{
		obj->g();
	}
}

int main()
{
	std::vector<std::unique_ptr<Base>> objs;
	objs.push_back(std::unique_ptr<Base>(new Sub1(1, 2)));
	objs.push_back(std::unique_ptr<Base>(new Sub2(3, 'd')));
	run(objs);
	cudaDeviceSynchronize();
	return 0;
}
