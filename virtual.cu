#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <utility>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

template<typename Base>
class device_copyable
{
public:
	virtual size_t getMostDerivedSize() const = 0;
	virtual std::pair<Base*, void*> placementNew(void* ptr) const = 0;
	virtual void copyFrom(void* ptr) = 0;
	virtual void resusciateOnDevice(void** pos, int len) = 0;
	virtual ~device_copyable() = default;
};

template<typename Derived, bool resuscitateVirtualBases>
__global__ void resuscitate_kernel(Derived** objects, int len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_count = gridDim.x * blockDim.x;
	for (int i=tid; i<len; i += thread_count)
	{
		if (resuscitateVirtualBases == false)
		{
			Derived s(*objects[i]);
			new (objects[i]) Derived(s);
		}
		else
		{
			new (objects[i]) Derived; // Assumes that the default constructor does not initialize data
		}
	}
}

template<typename Base, typename Derived, bool resuscitateVirtualBases=false>
class implements_device_copyable : virtual public device_copyable<Base>
{
public:
	size_t getMostDerivedSize() const override
	{
		return sizeof(Derived);
	}
	std::pair<Base*, void*> placementNew(void* ptr) const override
	{
		new (ptr) Derived(static_cast<const Derived&>(*this));
		Derived* dp = reinterpret_cast<Derived*>(ptr);
		return std::make_pair(static_cast<Base*>(dp), (void*)dp);
	}
	// Re-forms the object in the host. Changes the vtable to that of the host.
	void copyFrom(void* ptr) override
	{
		if (resuscitateVirtualBases == false)
		{
			new (static_cast<Derived*>(this)) Derived(*reinterpret_cast<Derived*>(ptr));
		}
		else
		{
			memcpy(static_cast<Derived*>(this), ptr, getMostDerivedSize());	// We cannot call the copy constructor, because it will most likely try to read data fields from a virtual base class and crash, since the vtable belongs to the device at this point.
											// So we copy the bytes of the object and run a default constructor on it to changed the vtables to the host's ones.
			new (static_cast<Derived*>(this)) Derived; // Assumes that the default constructor does not initialize data
		}
	}
	// Re-forms the object in the device. Changes the vtable to that of the device.
	void resusciateOnDevice(void** pos, int len) override
	{
		const int TB_SIZE = 128;
		const int THREAD_COUNT = 16384;
		int grid_size = (THREAD_COUNT+TB_SIZE-1)/TB_SIZE;
		resuscitate_kernel<Derived, resuscitateVirtualBases><<<grid_size, TB_SIZE>>>(reinterpret_cast<Derived**>(pos), len);
	}
};

class Base : public virtual device_copyable<Base>
{
public:
	int b_id;
public:
	Base() = default;
	__host__ __device__ Base(int _b_id):b_id(_b_id){}
	__host__ __device__ Base(const Base& b):b_id(b.b_id){}
	virtual ~Base() = default;
	__device__ virtual void f() = 0;
};

class Sub1 : public Base, public implements_device_copyable<Base, Sub1>
{
public:
	int sub_id;
public:
	Sub1() = default;
	__host__ __device__ Sub1(int _b_id, int _sub_id):Base(_b_id), sub_id(_sub_id){}
	__host__ __device__ Sub1(const Sub1& s):Base(s), sub_id(s.sub_id){}
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
	Sub2() = default;
	__host__ __device__ Sub2(int _b_id, char _sub_ch):Base(_b_id), sub_ch(_sub_ch){}
	__host__ __device__ Sub2(const Sub2& s):Base(s), sub_ch(s.sub_ch){}
	__device__ void f() override
	{
		printf("hello 2: %d %c!\n", b_id, sub_ch);
		sub_ch = 'Z';
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
	// Calculate the total size required for the unified memory buffer
	size_t totalSize=0, currentSize=0;
	for (const auto& obj : objs)
		totalSize += obj->getMostDerivedSize();
	// Group objects of the same dynamic type together
	std::unordered_map<std::type_index, std::vector<Base*>> groups;
	for (auto&& obj : objs)
	{
		groups[typeid(*obj)].push_back(obj.get());
	}
	// Allocate memory to store pointers to the objects
	void** d_objects_derived;
	Base** d_objects_base;
	int d_objects_index=0;
	CUDA_CALL(cudaMallocManaged((void **)&d_objects_derived, objs.size() * sizeof(void*)));
	CUDA_CALL(cudaMallocManaged((void **)&d_objects_base, objs.size() * sizeof(Base*)));
	// Allocate memory to store the objects
	char* object_buffer;
	CUDA_CALL(cudaMallocManaged((void **)&object_buffer, totalSize));
	// Copy the objects into unified memory
	for (const auto& pr : groups)
	{
		for (const auto& obj : pr.second)
		{
			auto ptrp = obj->placementNew(object_buffer + currentSize);
			d_objects_base[d_objects_index] = ptrp.first;
			d_objects_derived[d_objects_index++] = ptrp.second;
			currentSize += obj->getMostDerivedSize();
		}
	}
	// Resuscitate the objects, one group at a time.
	// Resuscitation lets the device code use the virtual functions of the objects (it probably fixes up the vtable? See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions)
	// We actually have a dedicated kernel for each dynamic type, which is what device_copyable::resusciate ends up running.
	// This keeps us from having to maintain RTTI manually. It also reduces branch divergence on the device.
	d_objects_index = 0;
	for (const auto& pr : groups)
	{
		pr.second[0]->resusciateOnDevice(&d_objects_derived[d_objects_index], pr.second.size());
		d_objects_index += pr.second.size();
	}
	// Run the demo to make sure everything works.
	runner<<<1,1>>>(d_objects_base, objs.size());
	// Synchronize with the device
	cudaDeviceSynchronize();
	// Copy the objects back to the host
	d_objects_index = 0;
	for (const auto& pr : groups)
	{
		for (const auto& obj : pr.second)
		{
			obj->copyFrom(d_objects_derived[d_objects_index++]);
		}
	}
	// Another demo to make sure that the changes come back to the host
	for (const auto& pr : groups)
	{
		for (const auto& obj : pr.second)
		{
			if (dynamic_cast<Sub2*>(obj))
				std::cout << static_cast<Sub2*>(obj)->sub_ch << std::endl;
		}
	}
	// Free all memory
	cudaFree((void*)d_objects_derived);
	cudaFree((void*)d_objects_base);
	cudaFree((void*)object_buffer);
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
