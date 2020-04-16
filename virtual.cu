#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

enum class TypeId // CUDA does not support device-side RTTI
{ //TODO: Can we get rid of this?
	Sub1, Sub2
};

template<typename Base>
class device_copyable
{
public:
	virtual size_t getMostDerivedSize() const = 0;
	virtual Base* placementNew(void* ptr) const = 0;
	//virtual void copyToDevice(Base** pos) = 0;
	virtual ~device_copyable() = default;
};

/*template<typename Base, typename Derived>
__global__ void copy_to_device(Base** object, Derived h)
{
	*object = new Derived(h);
}*/

template<typename Base, typename Derived>
class implements_device_copyable : virtual public device_copyable<Base>
{
public:
	size_t getMostDerivedSize() const override
	{
		return sizeof(Derived);
	}
	Base* placementNew(void* ptr) const override
	{
		new (ptr) Derived(static_cast<const Derived&>(*this));
		Derived* dp = reinterpret_cast<Derived*>(ptr);
		return static_cast<Base*>(dp);
	}
	/*void copyToDevice(Base** pos) override
	{
		copy_to_device<Base, Derived><<<1,1>>>(pos, static_cast<Derived&>(*this));
	}*/
};

class Base : public virtual device_copyable<Base>
{
public:
	TypeId type_id;
	int b_id;
public:
	__host__ __device__ Base(TypeId _type_id, int _b_id):type_id(_type_id), b_id(_b_id){}
	__host__ __device__ Base(TypeId _type_id, const Base& b):type_id(_type_id), b_id(b.b_id){}
	virtual ~Base() = default;
	__device__ virtual void f() = 0;
};

class Sub1 : public Base, public implements_device_copyable<Base, Sub1>
{
public:
	int sub_id;
public:
	__host__ __device__ Sub1(int _b_id, int _sub_id):
		Base(TypeId::Sub1, _b_id), sub_id(_sub_id)
	{}
	__host__ __device__ Sub1(const Sub1& s):
		Base(TypeId::Sub1, s), sub_id(s.sub_id)
	{}
	__device__ void f() override
	{
		printf("hello 1: %d %d!\n", b_id, sub_id);
	}
};

class Sub2 : public Base, public implements_device_copyable<Base, Sub2>
{
public:
	char sub_ch;
public:
	__host__ __device__ Sub2(int _b_id, char _sub_ch):
		Base(TypeId::Sub1, _b_id), sub_ch(_sub_ch)
	{}
	__host__ __device__ Sub2(const Sub2& s):
		Base(TypeId::Sub2, s), sub_ch(s.sub_ch)
	{}
	__device__ void f() override
	{
		printf("hello 2: %d %c!\n", b_id, sub_ch);
	}
};

__global__ void resuscitate(Base** objects, int len)
{
	for (int i=0; i<len; i++)
	{
		// TODO: Do not resolve RTTI on device side, as this will result in high divergence (UNLESS objects of the same type are next to one another)
		//  Instead, resolve them on host side(how?)
		//    Have a correcponsing resusciate for each MDT (template resusciate). Make it into a virtual function in device_copyable. In *run*, group objects by their MDT, and for each group, call its virtual resusciate, which'll in turn call the templated resusciate_kernel tailored for that BTC.
		if (objects[i]->type_id == TypeId::Sub1)
		{
			Sub1 s(*static_cast<Sub1*>(objects[i]));
			new (objects[i]) Sub1(s);
		}
		else if (objects[i]->type_id == TypeId::Sub2)
		{
			Sub2 s(*static_cast<Sub2*>(objects[i]));
			new (objects[i]) Sub2(s);
		}
	}
}


__global__ void runner(Base** objects, int len)
{
	for (int i=0; i<len; i++)
	{
		objects[i]->f();
	}
}

void run(std::vector<std::unique_ptr<Base>> objs)
{
	Base** d_objects;
	CUDA_CALL(cudaMallocManaged((void **)&d_objects, objs.size() * sizeof(Base*)));
	size_t totalSize=0, currentSize=0;
	for (const auto& obj : objs)
		totalSize += obj->getMostDerivedSize();
	char* object_buffer;
	CUDA_CALL(cudaMallocManaged((void **)&object_buffer, totalSize));
	for (int i=0; i<objs.size(); i++)
	{
		d_objects[i] = objs[i]->placementNew(object_buffer + currentSize);
		currentSize += objs[i]->getMostDerivedSize();
	}
	resuscitate<<<1,1>>>(d_objects, objs.size());
	runner<<<1,1>>>(d_objects, objs.size());
}

int main()
{
	std::vector<std::unique_ptr<Base>> objs;
	objs.push_back(std::unique_ptr<Base>(new Sub1(1, 2)));
	objs.push_back(std::unique_ptr<Base>(new Sub2(3, 'd')));
	run(std::move(objs));
	cudaDeviceSynchronize();
	return 0;
}
