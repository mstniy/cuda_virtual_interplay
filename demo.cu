#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>
#include "virtual_interplay.h"

#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)
#endif

using namespace std;

class Base : public Managed
{
public:
	int b_id;
public:
	Base() = default;
	__host__ __device__ Base(int _b_id):b_id(_b_id){}
	virtual ~Base() = default;
	__device__ virtual void f() = 0;
	__host__ __device__ virtual void g() = 0;
};

// Classes with virtual bases are supported
class Sub1 : virtual public Base
{
public:
	int sub_id;
public:
	Sub1() = default;
	__host__ __device__ Sub1(int _b_id, int _sub_id):Base(_b_id), sub_id(_sub_id){}
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

class Sub2 : public Base
{
public:
	char sub_ch;
public:
	Sub2() = delete; // non-default constructible classes are also supported
	__host__ __device__ Sub2(int _b_id, char _sub_ch):Base(_b_id), sub_ch(_sub_ch){}
	Sub2(const Sub2&) = delete; // non-copy constructible classes are also supported
	Sub2(Sub2&& s) = default;
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

__global__ void runner(Unified<Base*>* objects, int len)
{
	for (int i=0; i<len; i++)
	{
		objects[i]->f();
	}
}

int main()
{
	unique_ptr<Sub1> s1(new Sub1(1, 2));
	unique_ptr<Sub2> s2(new Sub2(3, 'd'));
	ClassMigrator<Sub1, true> sub1_migrator(s1.get(), 1);
	ClassMigrator<Sub2> sub2_migrator(s2.get(), 1);
	unique_ptr<Unified<Base*>[]> objs(new Unified<Base*>[2]);
	objs[0] = s1.get();
	objs[1] = s2.get();
	// Migrate the objects to the device
	sub1_migrator.toDevice();
	sub2_migrator.toDevice();
	cudaDeviceSynchronize();
	// Run the demo to make sure everything works.
	runner<<<1,1>>>(objs.get(), 2);
	// Migrate the objects back to the host
	sub1_migrator.toHost();
	sub2_migrator.toHost();
	// Another demo to make sure that the changes come back to the host
	objs[0]->g();
	objs[1]->g();
	cudaDeviceSynchronize();
	return 0;
}
