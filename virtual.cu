#include <stdio.h>
#include <vector>
#include <memory>
#include <iostream>
#include <typeindex>
#include <unordered_map>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

using namespace std;

template<typename Base>
class device_copyable
{
public:
	virtual size_t getMostDerivedSize() const = 0;
	virtual Base* placementNew(void* ptr) const = 0;
	virtual void resusciate(void** pos, int len) = 0;
	virtual ~device_copyable() = default;
};

template<typename Derived>
__global__ void resuscitate_kernel(Derived** objects, int len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_count = gridDim.x * blockDim.x;
	for (int i=tid; i<len; i += thread_count)
	{
		Derived s(*static_cast<Derived*>(objects[i]));
		new (objects[i]) Derived(s);
	}
}

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
	void resusciate(void** pos, int len) override
	{
		const int TB_SIZE = 128;
		const int THREAD_COUNT = 16384;
		int grid_size = (THREAD_COUNT+TB_SIZE-1)/TB_SIZE;
		resuscitate_kernel<Derived><<<grid_size, TB_SIZE>>>(reinterpret_cast<Derived**>(pos), len);
		//cudaDeviceSynchronize();
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
	__device__ virtual void f() = 0;
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

class Sub2 : public Base, public implements_device_copyable<Base, Sub2>
{
public:
	char sub_ch;
public:
	__host__ __device__ Sub2(int _b_id, char _sub_ch):Base(_b_id), sub_ch(_sub_ch){}
	__host__ __device__ Sub2(const Sub2& s):Base(s), sub_ch(s.sub_ch){}
	__device__ void f() override
	{
		printf("hello 2: %d %c!\n", b_id, sub_ch);
	}
};

__global__ void runner(Base** objects, int len)
{
	for (int i=0; i<len; i++)
	{
		objects[i]->f();
	}
}

void run(std::vector<std::unique_ptr<Base>> objs)
{
	// Calculate the total size required for the unified memory buffer
	size_t totalSize=0, currentSize=0;
	for (const auto& obj : objs)
		totalSize += obj->getMostDerivedSize();
	// Group objects of the same dynamic type together
	std::unordered_map<std::type_index, std::vector<std::unique_ptr<Base>>> groups;
	for (auto&& obj : objs)
	{
		groups[typeid(*obj)].push_back(std::move(obj));
	}
	// Allocate memory to store pointers to the objects
	Base** d_objects;
	int d_objects_index=0;
	CUDA_CALL(cudaMallocManaged((void **)&d_objects, objs.size() * sizeof(Base*)));
	// Allocate memory to store the objects
	char* object_buffer;
	CUDA_CALL(cudaMallocManaged((void **)&object_buffer, totalSize));
	// Copy the objects into unified memory
	for (const auto& pr : groups)
	{
		for (const auto& obj : pr.second)
		{
			d_objects[d_objects_index++] = obj->placementNew(object_buffer + currentSize);
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
		pr.second[0]->resusciate((void**)&d_objects[d_objects_index], pr.second.size());
		d_objects_index += pr.second.size();
	}
	// Run the demo to make sure everything works.
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
