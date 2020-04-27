#ifndef VIRTUAL_INTERPLAY_H
#define VIRTUAL_INTERPLAY_H

#include <stdio.h>
#include <type_traits>
#include <new>
#include "cuda_memory"

template<typename T>
__global__ void construct_object_kernel(T* ptr)
{
	new (ptr) T;
}

template<typename T>
__global__ void destruct_object_kernel(T* ptr)
{
	ptr->~T();
}

template<typename T>
struct my_aligned_storage {
	alignas(T) unsigned char data[sizeof(T)];
};

template<typename T>
class Resuscitator
{
public:
	my_aligned_storage<T> d_t_buffer={0}, h_t_buffer={0}; // A pair of T-s, one is constructed on the device, the other on the host. Initialized with zeros in case the constructor leaves some data fields uninitialized.
	// Bytemap describing which bytes of T belong to vtable and which ones ara part of data
	// vtable bytes get corrected during resuscitation, data bytes do not get overwritten.
	bool isPartOfVtable[sizeof(T)]={0};
public:
	Resuscitator()
	{
		new (&h_t_buffer) T;
		construct_object_kernel<T><<<1,1>>>((T*)&d_t_buffer); // This is fine because Resuscitator is always created in unified memory
		cudaDeviceSynchronize();
		for (size_t i=0; i<sizeof(T);i++)
		{
			isPartOfVtable[i] = d_t_buffer.data[i] != h_t_buffer.data[i];
		}
	}

	__host__ __device__ void resuscitate(T* object) const
	{
		for (size_t i=0; i<sizeof(T); i++)
			if (isPartOfVtable[i])
			{
#ifdef __CUDA_ARCH__
				((char*)object)[i] = d_t_buffer.data[i];
#else
				((char*)object)[i] = h_t_buffer.data[i];
#endif
			}
	}

	~Resuscitator()
	{
		((T*)&h_t_buffer)->~T();
		destruct_object_kernel<<<1, 1>>>((T*)&d_t_buffer);
		cudaDeviceSynchronize();
	}
};

template<typename T, bool resuscitateVirtualBases>
__global__ void resuscitate_kernel(T* objects, int len, const Resuscitator<T>* rtor)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_count = gridDim.x * blockDim.x;
	for (int i=tid; i<len; i += thread_count)
	{
		if (resuscitateVirtualBases)
			rtor->resuscitate(&objects[i]);
		else
		{
			T s(std::move(objects[i]));
			new (&objects[i]) T(std::move(s));
		}
	}
}

// It is amazing how many hoops you have to jump through to emulate "if constexpr" < C++17
template<typename T, bool resuscitateVirtualBases>
class CreateResuscitatorImpl;

template<typename T>
class CreateResuscitatorImpl<T, false>
{
public:
	static unified_unique_ptr<Resuscitator<T>> create()
	{
		// We do not need Resuscitator if we do not need to resuscitate virtual bases
		return {};
	}
};

template<typename T>
class CreateResuscitatorImpl<T, true>
{
public:
	static unified_unique_ptr<Resuscitator<T>> create()
	{
		return make_unified_unique<Resuscitator<T>>();
	}
};


template<typename T, bool resuscitateVirtualBases=false>
class ClassMigrator
{
private:
	const int TB_SIZE = 128;
	const int THREAD_COUNT = 16384;

	unified_unique_ptr<Resuscitator<T>> rtor;
	T* objs;
	size_t length;
public:
	ClassMigrator() = default;
	ClassMigrator(const ClassMigrator&) = delete;
	ClassMigrator(ClassMigrator&& o):
		rtor(o.rtor),
		objs(o.objs),
		length(o.length)
	{
		o.rtor = NULL;
	}
	ClassMigrator(T* _objs, size_t _length):
		objs(_objs),
		length(_length)
	{
		if (length == 0)
			return ;

		rtor = CreateResuscitatorImpl<T, resuscitateVirtualBases>::create();
	}

	void toDevice()
	{
		// Resuscitate the objects.
		// Resuscitation lets the device use the virtual functions and access the virtual bases of objects created on the host and vica versa.
		// We actually have a dedicated kernel for each dynamic type, which is what interface_movable::resuscitateOnDevice ends up running.
		// This keeps us from having to maintain RTTI manually. It also reduces branch divergence on the device.
		int grid_size = (THREAD_COUNT+TB_SIZE-1)/TB_SIZE;
		resuscitate_kernel<T, resuscitateVirtualBases><<<grid_size, TB_SIZE>>>(objs, length, rtor.get());
		cudaDeviceSynchronize();
	}

	void toHost() // Careful: overwrites the given objects
	{
		cudaDeviceSynchronize();
		// Re-construct the objects on the host
		for (size_t i=0; i<length; i++)
		{
			if (resuscitateVirtualBases)
			{
				rtor->resuscitate(&objs[i]); // We need to resuscitate the object because the move constructor will probably access the virtual bases and crash
			}
			else // If we do not need to resuscitate virtual bases, a simple move construction is enough to fix the (single) vtable.
			{
				T temp{std::move(objs[i])};
				// We cannot call the destructor of objs[i] here, because it is most likely virtual thus the call will crash.
				new (&objs[i]) T(std::move(temp));
			}
		}
	}
};

#endif
