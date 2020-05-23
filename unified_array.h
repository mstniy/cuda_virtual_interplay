#ifndef UNIFIED_ARRAY_H
#define UNIFIED_ARRAY_H

#include "cuda.h"
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <type_traits>
#include <memory>

template<typename T>
class UnifiedArrayView
{
protected:
	T* arr=NULL;
	size_t _length=0;
protected:
	UnifiedArrayView(T* _arr, size_t __length):
		arr(_arr),
		_length(__length)
	{
	}
public:
	UnifiedArrayView() = default;

	T* data()
	{
		return arr;
	}

	const T* data() const
	{
		return arr;
	}

	__host__ __device__ T& operator[](int index)
	{
		return arr[index];
	}

	__host__ __device__ const T& operator[](int index) const
	{
		return arr[index];
	}

	__host__ __device__ size_t length() const
	{
		return _length;
	}

	__host__ __device__ T* begin()
	{
		return arr;
	}

	__host__ __device__ const T* begin() const
	{
		return arr;
	}

	__host__ __device__ T* end()
	{
		return arr+_length;
	}

	__host__ __device__ const T* end() const
	{
		return arr+_length;
	}
};

// A typed array of fixed size in unified-memory
// Notice that although the underlying buffer that holds T-s is in unified memory, the UnifiedArray itself also needs to be allocated in unified memory to let the device interact with UnifiedArray itself.
// To let the device operate on UnifiedArray-s, pass UnifiedArrayView-s obtained by calling getView().
template<typename T>
class UnifiedArray : public UnifiedArrayView<T>
{
	using UnifiedArrayView<T>::arr;
	using UnifiedArrayView<T>::_length;
public:
	UnifiedArray() = default;
	UnifiedArray(size_t __length):
		UnifiedArrayView<T>(nullptr, __length)
	{
		if (_length == 0)
			return ;

		if (cudaSuccess != cudaMallocManaged((void**)&arr, sizeof(T)*_length))
			throw std::bad_alloc();
		cudaDeviceSynchronize();
		for (size_t i=0; i<_length; i++)
			new (&arr[i]) T;
	}

	UnifiedArray(const std::vector<T>& v):
		UnifiedArrayView<T>(nullptr, v.size())
	{
		if (_length == 0)
			return ;

		if (cudaSuccess != cudaMallocManaged((void**)&arr, sizeof(T)*_length))
			throw std::bad_alloc();
		cudaDeviceSynchronize();
		for (size_t i=0; i<_length; i++)
			new (&arr[i]) T(v[i]);
	}

	UnifiedArray(const UnifiedArray&) = delete;
	UnifiedArray& operator=(const UnifiedArray&) = delete;

	UnifiedArray(UnifiedArray&& o):
		UnifiedArrayView<T>(o.arr, o._length)
	{
		o.arr = NULL;
	}

	UnifiedArray& operator=(UnifiedArray&& o)
	{
		if (&o == this)
			return (*this);

		this->~UnifiedArray();
		new (this) UnifiedArray(std::move(o));
		return (*this);
	}

	~UnifiedArray()
	{
		if (arr != NULL)
		{
			cudaDeviceSynchronize();
			for (size_t i=0; i<_length; i++)
				arr[i].~T();
			cudaFree(arr);
		}
	}

	UnifiedArrayView<T> getView()
	{
		return *this;
	}

	const UnifiedArrayView<T> getView() const
	{
		return *this;
	}
};

#endif
