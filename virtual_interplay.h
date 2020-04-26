#ifndef VIRTUAL_INTERPLAY_H
#define VIRTUAL_INTERPLAY_H

#include <stdio.h>
#include <type_traits>
#include <new>

class Managed
{
public:
	void* operator new(std::size_t count, const std::nothrow_t& /*tag*/) noexcept
	{
		void* ptr;
		if (cudaSuccess != cudaMallocManaged(&ptr, count))
			return nullptr;
		cudaDeviceSynchronize();
		return ptr;
	}
	__host__ __device__ void* operator new(std::size_t count)
	{
#ifdef __CUDA_ARCH__
		return ::operator new(count);
#else
		void* ptr = operator new(count, std::nothrow);
		if (ptr == nullptr)
			throw std::bad_alloc{};
		return ptr;
#endif
	}
	__host__ __device__ void* operator new(std::size_t /*count*/, void* ptr)
	{
		return ptr;
	}
	void* operator new[](std::size_t count, const std::nothrow_t& tag) noexcept
	{
		return operator new(count, tag);
	}
	__host__ __device__ void* operator new[](std::size_t count)
	{
#ifdef __CUDA_ARCH__
		return ::operator new[](count);
#else
		return operator new(count);
#endif
	}
	__host__ __device__ void* operator new[](std::size_t /*count*/, void* ptr)
	{
		return ptr;
	}
	__host__ __device__ void operator delete(void* ptr)
	{
#ifdef __CUDA_ARCH__
		::operator delete(ptr);
#else
		operator delete(ptr, std::nothrow);
#endif
	}
	void operator delete(void* ptr, const std::nothrow_t& /*tag*/) noexcept
	{
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
	__host__ __device__ void operator delete(void* /*ptr*/, void* /*place*/)
	{
	}
	void operator delete[](void* ptr, const std::nothrow_t& /*tag*/) noexcept
	{
		operator delete(ptr);
	}
	__host__ __device__ void operator delete[](void* ptr)
	{
#ifdef __CUDA_ARCH__
		::operator delete[](ptr);
#else
		operator delete(ptr);
#endif
	}
	__host__ __device__ void operator delete[](void* /*ptr*/, void* /*place*/)
	{
	}
};

template<typename T>
class Unified : public Managed
{
public:
	T t;
public:
	Unified() = default;
	Unified(const Unified&) = default;
	Unified(Unified&&) = default;
	Unified(const T& _t): t(_t){}
	Unified(T&& _t): t(std::move(_t)){}
	Unified& operator=(const Unified&) = default;
	Unified& operator=(Unified&&) = default;
	__host__ __device__ Unified& operator=(const T& _t){t=_t; return *this;}
	__host__ __device__ Unified& operator=(T&& _t){t=std::move(_t); return *this;}
	__host__ __device__ T operator->() const {return t;}
	__host__ __device__ typename std::add_lvalue_reference<typename std::remove_pointer<T>::type>::type operator*() const {return *t;}
	__host__ __device__ typename std::add_lvalue_reference<typename std::remove_pointer<T>::type>::type operator[](size_t index) const {return t[index];}
	__host__ __device__ operator T() const {return t;}
};

template<typename T>
__global__ void construct_object_kernel(T* ptr)
{
	memset(ptr, 0, sizeof(T)); // Initialize all bytes to 0 in case the constructor leaves some data fields uninitialized
	new (ptr) T;
}

template<typename T>
__global__ void destruct_object_kernel(T* ptr)
{
	ptr->~T();
}

template<typename T>
class Resuscitator
{
public:
	T* d_t; // A T which is constructed on the device
	T* h_t; // A T which is constructed on the host
	// Bytemap describing which bytes of T belong to vtable and which ones ara part of data
	// vtable bytes get corrected during resuscitation, data bytes do not get overwritten.
	bool isPartOfVtable[sizeof(T)]={};
public:
	Resuscitator()
	{
		h_t = (T*)malloc(sizeof(T));
		memset(h_t, 0, sizeof(T)); // Initialize all bytes to 0 in case the constructor leaves some data fields uninitialized
		new (h_t) T;
		if (cudaSuccess != cudaMallocManaged((void **)&d_t, sizeof(T)))
			throw std::bad_alloc();
		construct_object_kernel<T><<<1,1>>>(d_t);
		cudaDeviceSynchronize();
		for (size_t i=0; i<sizeof(T);i++)
		{
			isPartOfVtable[i] = ((char*)d_t)[i] != ((char*)h_t)[i];
		}
	}

	__host__ __device__ void resuscitate(T* object) const
	{
		for (size_t i=0; i<sizeof(T); i++)
			if (isPartOfVtable[i])
			{
#ifdef __CUDA_ARCH__
				((char*)object)[i] = ((char*)d_t)[i];
#else
				((char*)object)[i] = ((char*)h_t)[i];
#endif
			}
	}

	~Resuscitator()
	{
		h_t->~T();
		free(h_t);
		destruct_object_kernel<<<1, 1>>>(d_t);
		cudaDeviceSynchronize();
		cudaFree(d_t);
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
	static Resuscitator<T>* create()
	{
		// We do not need Resuscitator if we do not need to resuscitate virtual bases
		return NULL;
	}
};

template<typename T>
class CreateResuscitatorImpl<T, true>
{
public:
	static Resuscitator<T>* create()
	{
		Resuscitator<T>* rtor;
		if (cudaSuccess != cudaMallocManaged((void **)&rtor, sizeof(Resuscitator<T>)))
			throw std::bad_alloc();
		new (rtor) Resuscitator<T>;
		return rtor;
	}
};


template<typename T, bool resuscitateVirtualBases=false>
class ClassMigrator
{
private:
	const int TB_SIZE = 128;
	const int THREAD_COUNT = 16384;

	Resuscitator<T>* rtor = NULL;
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

	~ClassMigrator()
	{
		if (rtor != NULL)
		{
			rtor->~Resuscitator<T>();
			cudaFree((void*)rtor);
		}
	}

	void toDevice()
	{
		// Resuscitate the objects.
		// Resuscitation lets the device use the virtual functions and access the virtual bases of objects created on the host and vica versa.
		// We actually have a dedicated kernel for each dynamic type, which is what interface_movable::resuscitateOnDevice ends up running.
		// This keeps us from having to maintain RTTI manually. It also reduces branch divergence on the device.
		int grid_size = (THREAD_COUNT+TB_SIZE-1)/TB_SIZE;
		resuscitate_kernel<T, resuscitateVirtualBases><<<grid_size, TB_SIZE>>>(objs, length, rtor);
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
