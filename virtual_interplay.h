#ifndef VIRTUAL_INTERPLAY_H
#define VIRTUAL_INTERPLAY_H

#include <stdio.h>
#include <vector>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)
#endif

template<typename Derived>
__global__ void construct_object_kernel(Derived* ptr)
{
	memset(ptr, 0, sizeof(Derived)); // Initialize all bytes to 0 in case the constructor leaves some data fields uninitialized
	new (ptr) Derived;
}

template<typename Derived>
__global__ void destruct_object_kernel(Derived* ptr)
{
	ptr->~Derived();
}

template<typename Derived>
class Resuscitator
{
public:
	Derived* d_derived; // A Derived which is constructed on the device
	Derived* h_derived; // A Derived which is constructed on the host
	// Bytemap describing which bytes of Derived belong to vtable and which ones ara part of data
	// vtable bytes get corrected during resuscitation, data bytes do not get overwritten.
	bool isPartOfVtable[sizeof(Derived)]={};
public:
	Resuscitator()
	{
		h_derived = (Derived*)malloc(sizeof(Derived));
		memset(h_derived, 0, sizeof(Derived)); // Initialize all bytes to 0 in case the constructor leaves some data fields uninitialized
		new (h_derived) Derived;
		CUDA_CALL(cudaMallocManaged((void **)&d_derived, sizeof(Derived)));
		construct_object_kernel<Derived><<<1,1>>>(d_derived);
		cudaDeviceSynchronize();	
		for (int i=0; i<sizeof(Derived);i++)
		{
			isPartOfVtable[i] = ((char*)d_derived)[i] != ((char*)h_derived)[i];
		}
	}

	__host__ __device__ void resuscitate(Derived* object) const
	{
		for (int i=0; i<sizeof(Derived); i++)
			if (isPartOfVtable[i])
			{
#ifdef __CUDA_ARCH__
				((char*)object)[i] = ((char*)d_derived)[i];
#else
				((char*)object)[i] = ((char*)h_derived)[i];
#endif
			}
	}

	~Resuscitator()
	{
		h_derived->~Derived();
		free(h_derived);
		destruct_object_kernel<<<1, 1>>>(d_derived);
		cudaDeviceSynchronize();
		cudaFree(d_derived);
	}
};

template<typename Derived>
__global__ void resuscitate_kernel(Derived** objects, int len, const Resuscitator<Derived>* rtor)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_count = gridDim.x * blockDim.x;
	for (int i=tid; i<len; i += thread_count)
	{
		rtor->resuscitate(objects[i]);
	}
}

template<typename Base>
class interplay_movable
{
public:
	virtual size_t getMostDerivedSize() const = 0;
	virtual const void* createResuscitator() const = 0;
	virtual void deleteResuscitator(const void* rtor) const = 0;
	virtual Base* moveTo(void* ptr) = 0;
	virtual void moveFrom(void* ptr, const void* rtor) = 0;
	virtual void resuscitateOnDevice(void** pos, int len, const void* rtor) = 0;
	__host__ __device__ virtual ~interplay_movable(){};
};

template<typename Base, typename Derived>
class implements_interplay_movable : virtual public interplay_movable<Base>
{
public:
	size_t getMostDerivedSize() const override
	{
		return sizeof(Derived);
	}
	const void* createResuscitator() const override
	{
		Resuscitator<Derived>* rtor;
		CUDA_CALL(cudaMallocManaged((void **)&rtor, sizeof(Resuscitator<Derived>)));
		new (rtor) Resuscitator<Derived>;
		return rtor;
	}
	void deleteResuscitator(const void* rtor) const override
	{
		((Resuscitator<Derived>*)rtor)->~Resuscitator<Derived>();
		cudaFree((void*)rtor);
	}
	Base* moveTo(void* ptr) override
	{
		new (ptr) Derived(std::move(static_cast<Derived&>(*this)));
		Derived* dp = reinterpret_cast<Derived*>(ptr);
		return static_cast<Base*>(dp);
	}
	// Re-forms the object in the host. Changes the vtable to that of the host.
	void moveFrom(void* ptr, const void* rtor) override
	{
		((const Resuscitator<Derived>*)rtor)->resuscitate((Derived*)ptr); // We need to resuscitate the object in case it contains virtual bases
		this->~implements_interplay_movable();
		new (static_cast<Derived*>(this)) Derived(std::move(*(Derived*)(ptr)));
	}
	// Re-forms the object in the device. Changes the vtable to that of the device.
	void resuscitateOnDevice(void** pos, int len, const void* rtor) override
	{
		const int TB_SIZE = 128;
		const int THREAD_COUNT = 16384;
		int grid_size = (THREAD_COUNT+TB_SIZE-1)/TB_SIZE;
		resuscitate_kernel<Derived><<<grid_size, TB_SIZE>>>(reinterpret_cast<Derived**>(pos), len, (const Resuscitator<Derived>*)rtor);
	}
};

template<typename Base>
class ClassMigrator
{
private:
	size_t totalSize=0;
	std::unordered_map<std::type_index, std::pair<const void*, std::vector<std::pair<int, Base*>>>> groups;
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
			groups[typeid(*objs[i])].second.push_back(std::make_pair(i, objs[i]));
		}
		for (auto& pr : groups)
			pr.second.first = pr.second.second[0].second->createResuscitator();
		// Allocate memory to store pointers to the objects
		CUDA_CALL(cudaMallocManaged((void **)&d_objects_derived, objs.size() * sizeof(void*)));
		CUDA_CALL(cudaMallocManaged((void **)&d_objects_base, objs.size() * sizeof(Base*)));
		// Allocate memory to store the objects
		CUDA_CALL(cudaMallocManaged((void **)&object_buffer, totalSize));
	}

	~ClassMigrator()
	{
		for (auto& pr : groups)
			pr.second.second[0].second->deleteResuscitator(pr.second.first);
		cudaFree((void*)d_objects_derived);
		cudaFree((void*)object_buffer);
		cudaFree((void*)d_objects_base);
	}

	Base** toDevice()
	{
		int d_objects_index = 0;
		size_t currentSize=0;
		// Move the objects into unified memory
		for (const auto& pr : groups)
		{
			for (const auto& obj : pr.second.second)
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
		// We actually have a dedicated kernel for each dynamic type, which is what interface_movable::resuscitateOnDevice ends up running.
		// This keeps us from having to maintain RTTI manually. It also reduces branch divergence on the device.
		d_objects_index = 0;
		for (const auto& pr : groups)
		{
			pr.second.second[0].second->resuscitateOnDevice(&d_objects_derived[d_objects_index], pr.second.second.size(), pr.second.first);
			d_objects_index += pr.second.second.size();
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
			const void* rtor = pr.second.first;
			for (const auto& obj : pr.second.second)
			{
				obj.second->moveFrom(d_objects_derived[d_objects_index++], rtor);
			}
		}
	}
};

#endif
