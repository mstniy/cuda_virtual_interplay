A small library to make virtual calls and field accesses across CUDA boundaries easier.

It is not that hard to pass C++ classes over CUDA, especially with unified memory.
Unfortunately however, only the host can call the virtual methods of the objects created on the host and vice versa.
See [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions).
The case is similar with virtual inheritance. Only the host can access the member fields and methods of the virtual bases of the objects created on the host and vice versa.
This library lets you smoothly migrate such objects from the host to the device and vice versa.
How? It basically copy-constructs the objects where they are needed, so they retain their data while having a valid vtable.

It requires that:
* Your GPU supports unified memory (everything above and including Kepler supports it)
* The classes to be migrated do not manage any resources (or they will leak)
* The classes to be migrated are copy-constructible

It can also migrate classes with virtual bases. In this case it also requires that the default constructor does not initialize any data field. So this won't work:

    class A  : virtual Base
    {
    public:
    	int x=0;
    	...
    };

But this will:

    class A : virtual Base
    {
    public:
    	int x;
    	A() = default;
    };

## Usage

`#include "virtual_interplay.h"` to access the library. The base class of the classes to be migrated shall inherit from `device_copyable` and the most derived subclasses shall inherit from `implements_device_copyable`.
Use `__host__ __device__` to make sure your copy constructor is callable from both the host and the device.
If you want to access virtual bases across CUDA boundaries, set `resuscitateVirtualBases=true` and make sure that the default constructor (and the default constructors of all the parent classes) do not initialize any data fields.
Mark the virtual functions with `__device__` to let them be used from the device, or with `__host__ __device__` to let them be used on both sides.

To migrate an array of objects to the device, do:

    ClassMigrator<Base> migrator(objs);
    migrator.toDevice();


To migrate them back to the host, use `migrator.toHost()`

For a complete example, see `demo.cu`
