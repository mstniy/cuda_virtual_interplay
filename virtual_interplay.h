#ifndef VIRTUAL_INTERPLAY_H
#define VIRTUAL_INTERPLAY_H

#include <stdio.h>
#include <vector>
#include <typeindex>
#include <unordered_map>
#include <utility>

#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)
#endif

template<typename Base>
class device_copyable
{
public:
	virtual size_t getMostDerivedSize() const = 0;
	virtual Base* moveTo(void* ptr) = 0;
	virtual void moveFrom(void* ptr) = 0;
	virtual void resusciateOnDevice(void** pos, int len) = 0;
	__host__ __device__ virtual ~device_copyable(){};
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
			Derived s(std::move(*objects[i]));
			new (objects[i]) Derived(std::move(s));
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
	Base* moveTo(void* ptr) override
	{
		new (ptr) Derived(std::move(static_cast<Derived&>(*this)));
		Derived* dp = reinterpret_cast<Derived*>(ptr);
		return static_cast<Base*>(dp);
	}
	// Re-forms the object in the host. Changes the vtable to that of the host.
	void moveFrom(void* ptr) override
	{
		this->~implements_device_copyable();

		if (resuscitateVirtualBases == false)
		{
			new (static_cast<Derived*>(this)) Derived(std::move(*reinterpret_cast<Derived*>(ptr)));
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

template<typename Base>
class ClassMigrator
{
private:
	size_t totalSize=0;
	std::unordered_map<std::type_index, std::vector<std::pair<int, Base*>>> groups;
	void** d_objects_derived=NULL;
	Base** d_objects_base=NULL;
	char* object_buffer=NULL;
public:
	ClassMigrator() = default;
	ClassMigrator(const ClassMigrator&) = delete;
	ClassMigrator(ClassMigrator&& o):
		totalSize(o.totalSize),
		groups(std::move(o.groups)),
		d_objects_derived(o.d_objects_derived),
		d_objects_base(o.d_objects_base),
		object_buffer(o.object_buffer)
	{
		o.d_objects_derived = NULL;
		o.object_buffer = NULL;
		o.d_objects_base=NULL;
	}
	ClassMigrator(std::vector<Base*> objs)
	{
		// Calculate the total size required for the unified memory buffer
		for (const auto& obj : objs)
			totalSize += obj->getMostDerivedSize();
		// Group objects of the same dynamic type together
		for (int i=0; i<objs.size(); i++)
		{
			groups[typeid(*objs[i])].push_back(std::make_pair(i, objs[i]));
		}
		// Allocate memory to store pointers to the objects
		CUDA_CALL(cudaMallocManaged((void **)&d_objects_derived, objs.size() * sizeof(void*)));
		CUDA_CALL(cudaMallocManaged((void **)&d_objects_base, objs.size() * sizeof(Base*)));
		// Allocate memory to store the objects
		CUDA_CALL(cudaMallocManaged((void **)&object_buffer, totalSize));
	}

	~ClassMigrator()
	{
		cudaFree((void*)d_objects_derived);
		cudaFree((void*)object_buffer);
		cudaFree((void*)d_objects_base);
	}

	Base** toDevice()
	{
		int d_objects_index = 0;
		size_t currentSize=0;
		// Copy the objects into unified memory
		for (const auto& pr : groups)
		{
			for (const auto& obj : pr.second)
			{
				void* derived_ptr = &object_buffer[currentSize];
				Base* base_ptr = obj.second->moveTo(derived_ptr);
				d_objects_base[obj.first] = base_ptr;
				d_objects_derived[d_objects_index++] = derived_ptr;
				currentSize += obj.second->getMostDerivedSize();
			}
		}
		// Resuscitate the objects, one group at a time.
		// Resuscitation lets the device code use the virtual functions of the objects (it probably fixes up the vtable? See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions)
		// We actually have a dedicated kernel for each dynamic type, which is what device_copyable::resusciate ends up running.
		// This keeps us from having to maintain RTTI manually. It also reduces branch divergence on the device.
		d_objects_index = 0;
		for (const auto& pr : groups)
		{
			pr.second[0].second->resusciateOnDevice(&d_objects_derived[d_objects_index], pr.second.size());
			d_objects_index += pr.second.size();
		}

		return d_objects_base;
	}

	void toHost() // Careful: overwrites the given objects
	{
		cudaDeviceSynchronize();
		// Copy the objects back to the host
		int d_objects_index = 0;
		for (const auto& pr : groups)
		{
			for (const auto& obj : pr.second)
			{
				obj.second->moveFrom(d_objects_derived[d_objects_index++]);
			}
		}
	}
};

#endif
